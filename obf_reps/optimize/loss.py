from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from obf_reps.logging import DummyLogger, Logger
from obf_reps.metrics import ObfMetric


class LossFunctionBase(ABC):

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float,
        logger: Logger = DummyLogger(),
    ):
        """
        Args:
            obf_metric: Metric to compute obfuscation
            lambda_: Weighting between behavior loss and obfuscation
                loss. Weighting is (lambda_) * behavior_loss and
                (1 - lambda_) * obfuscation_loss
        """
        self.obf_metric = obf_metric

        assert lambda_ >= 0 and lambda_ <= 1
        self.lambda_ = lambda_
        self.logger = logger

    @abstractmethod
    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss for each example in the batch.

        This function recieves lots of information and is designed to be as flexible as possible.
        Note that we only provie a mask over the target reps, this is because we can infer what the
        mask over the input reps would be as they are left padded, and we will only ever want to
        take loss over the last few tokens of the input, if ever.

        If needed, the infra can be adapted to provide input mask also.
        """
        ...


# ----- TWO TYPES OF LOSS FUNCTIONS ----- #
#
# **FlexibleLossFunctionBase**:
# Is not tied to a specific
# method of aggregating the obfuscation loss (i.e. using
# train_reps_reduce). This allows you to flexibly define
# how you aggregate the obfuscation loss from the metric
# values over the reps.
#
# **ProbePredictLoss**:
# This is tied to specific to the probe prediction.
# It directly uses probe.predict to get the obfuscation loss.
#
#
# If you want to explore how obfuscation can transfer between
# different probes or aggregation strategies,
# then use the flexible loss not attached to a specific predict
# method. If you want to just attack a specific probe for a
# specific task, don't mess around and just use the second one.


class ProbePredictLoss(LossFunctionBase):
    """The obfuscation loss is exactly what the probe predicts."""

    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss."""

        # Same as FlexibleLossFunctionBase.compute_loss
        b_size = behavior_logits.size(0)
        lambda_beh = self.lambda_
        lambda_obf = 1 - self.lambda_

        # Compute behavior loss
        beh_losses: Float[Tensor, "b_size beh_seq_len"] = F.cross_entropy(
            behavior_logits.transpose(1, 2), behavior_target, reduction="none"
        )

        beh_losses *= behavior_loss_mask.float()
        behavior_loss: Float[Tensor, "b"] = beh_losses.mean(dim=-1)
        assert behavior_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {behavior_loss.shape}"

        # Different to FlexibleLossFunctionBase.compute_loss
        reps, attention_mask = self.obf_metric.data_module.train_reps_reduce(
            input_reps=input_reps,
            target_reps=target_reps,
            target_mask=target_rep_loss_mask,
        )

        obf_loss: Float[Tensor, "b"] = self.obf_metric.predict(reps, attention_mask)
        assert obf_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {obf_loss.shape}"

        self.logger.log(
            {
                "behavior_loss": behavior_loss.mean().item(),
                "obf_loss": obf_loss.mean().item(),
            }
        )

        return lambda_beh * behavior_loss + lambda_obf * obf_loss


class FlexibleLossFunctionBase(LossFunctionBase, ABC):
    """This loss function allows you to aggregate what a probe / metric says about the reps of each
    token in a flexible manner."""

    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss."""

        b_size = behavior_logits.size(0)

        lambda_beh = self.lambda_
        lambda_obf = 1 - self.lambda_

        # Compute behavior loss
        beh_losses: Float[Tensor, "b_size beh_seq_len"] = F.cross_entropy(
            behavior_logits.transpose(1, 2), behavior_target, reduction="none"
        )

        beh_losses *= behavior_loss_mask.float()
        behavior_loss: Float[Tensor, "b"] = beh_losses.mean(dim=-1)
        assert behavior_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {behavior_loss.shape}"

        # Compute obfuscation loss
        obf_metrics: Float[Tensor, "b_size layers rep_seq_len"] = (
            self.obf_metric.evaluate(target_reps)
        )
        obf_loss: Float[Tensor, "b"] = self.aggregate_obf_loss(
            obf_metrics, target_rep_loss_mask
        )
        assert obf_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {obf_loss.shape}"

        self.logger.log(
            {
                "behavior_loss": behavior_loss.mean().item(),
                "obf_loss": obf_loss.mean().item(),
            }
        )

        return lambda_beh * behavior_loss + lambda_obf * obf_loss

    @abstractmethod
    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Aggregate the obfucation losses over the batch."""
        ...


class BehaviorLoss(FlexibleLossFunctionBase):
    """Just computes behavior loss irrespective of reps."""

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float = 0.5,
        logger: Logger = DummyLogger(),
    ):
        super().__init__(obf_metric=obf_metric, lambda_=1, logger=logger)

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        b_size = obf_metrics.size(0)
        return torch.zeros(b_size, device=obf_metrics.device)


class FinalTokenObfLoss(FlexibleLossFunctionBase):
    """Aggregates obfuscation loss from final token in sequences."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:

        # Get right most 1 from rep_loss_mask for each layer
        seq_len = obf_metrics.shape[2]
        num_layers = obf_metrics.shape[1]
        b_size = obf_metrics.shape[0]

        indices: Int64[Tensor, "b_size layers seq_len"] = torch.where(
            rep_loss_mask.unsqueeze(1).repeat(
                1, num_layers, 1
            ),  # Match rep_loss_mask to torch.arange
            torch.arange(seq_len, device=rep_loss_mask.device).expand_as(obf_metrics),
            -1,
        )

        rightmost_indices: Int64[Tensor, "b_size layers"] = indices.max(dim=2).values

        # Get the rightmost metric from obf_metrics
        rightmost_metrics: Float[Tensor, "b_size layers 1"] = torch.gather(
            obf_metrics, 2, rightmost_indices.unsqueeze(-1)
        )

        assert rightmost_metrics.shape == (
            b_size,
            num_layers,
            1,
        ), "Expected shape (b_size, layers, 1)"

        output = rightmost_metrics.mean(dim=1).squeeze()
        assert output.shape == (
            b_size,
        ), f"Expected shape (b_size,), got {output.shape}"

        return output


class MeanOverTokensObfLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to 0 where rep_loss_mask is False
        obf_metrics *= rep_loss_mask.unsqueeze(1).float()

        return obf_metrics.mean(dim=2).mean(dim=1)


class MaxOverTokensObfLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to 0 where rep_loss_mask is False
        obf_metrics = torch.where(
            rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf")
        )

        return obf_metrics.max(dim=2).values.mean(dim=1)


class MaxOverTokensAndLayersObfLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to -inf where rep_loss_mask is False
        obf_metrics = torch.where(
            rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf")
        )

        return obf_metrics.max(dim=2).values.max(dim=1).values


# ----- BitMap Loss Functions ----- #


class InputRobotLossFunction(LossFunctionBase):

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float = 0.5,
        logger: Logger = DummyLogger(),
    ):
        super().__init__(obf_metric=obf_metric, lambda_=lambda_, logger=logger)

        self.binary_mask: Float[Tensor, "layers seq_len"] = torch.from_numpy(
            np.load("./datasets/other/robot_binary.npy", allow_pickle=True)
        )

        num_layers, seq_len = self.binary_mask.shape
        assert num_layers == 32, "Only tested for 32 layers"

    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss."""

        b_size = behavior_logits.size(0)

        # If using larger batch sizes, would have to add a mask over input reps
        assert b_size == 1, "Only implemented for batch size 1"

        lambda_beh = self.lambda_
        lambda_obf = 1 - self.lambda_

        # Compute behavior loss
        beh_losses: Float[Tensor, "b_size beh_seq_len"] = F.cross_entropy(
            behavior_logits.transpose(1, 2), behavior_target, reduction="none"
        )

        beh_losses *= behavior_loss_mask.float()
        behavior_loss: Float[Tensor, "b"] = beh_losses.mean(dim=-1)
        assert behavior_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {behavior_loss.shape}"

        obf_metrics: Float[Tensor, "b_size layers rep_seq_len"] = (
            self.obf_metric.evaluate(input_reps)
        )
        input_seq_len = input_reps.shape[2]
        mask = torch.ones((1, input_seq_len), device=input_reps.device)
        obf_loss: Float[Tensor, "b"] = self.aggregate_obf_loss(obf_metrics, mask)

        assert obf_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {obf_loss.shape}"

        self.logger.log(
            {
                "behavior_loss": behavior_loss.mean().item(),
                "obf_loss": obf_loss.mean().item(),
            }
        )

        # Only return obf loss in this case
        return obf_loss

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "1 layers seq_len"],
        rep_loss_mask: Bool[Tensor, "1 seq_len"],
    ) -> Float[Tensor, "1"]:
        """Force the last parts of the input to look like the bit mask."""

        # Set metrics to -inf where rep_loss_mask is False
        # obf_metrics = torch.where(rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf"))
        b_size, num_layers, seq_len = obf_metrics.shape
        assert b_size == 1, "Only implemented for batch size 1"

        # Apply self.binary_mask to obf_metrics
        mask: Float[Tensor, "layers-1 len"] = self.binary_mask.to(obf_metrics.device)

        mask_num_layers, mask_seq_len = mask.shape

        # Pad with 0s on the bottom of dim 1 to match num_layers
        if mask_num_layers < num_layers:
            padding = torch.zeros(
                num_layers - mask_num_layers, mask_seq_len, device=mask.device
            )
            mask = torch.cat([mask, padding], dim=0)
        else:
            mask = mask[:num_layers, :]

        mask = mask.to(torch.bool)
        mask = mask.unsqueeze(0).repeat(b_size, 1, 1)

        label = ~mask
        # Cut scores to length of mask by taking last mask_seq_len elements
        assert seq_len >= mask_seq_len + 5
        obf_metrics = obf_metrics[:, :, -mask_seq_len - 5 : -5]
        # Expand label to match obf_metrics shape
        label = label.flip(0).float()
        label = label.flip(1)

        # Get MSE between obf_metrics and label, keeping batch dimension
        loss = (obf_metrics - label) ** 2
        output = loss.mean(dim=(1, 2))

        assert output.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {output.shape}"

        return output


class PinkElephantPatternLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float = 0.5,
        logger: Logger = DummyLogger(),
    ):
        super().__init__(obf_metric=obf_metric, lambda_=lambda_, logger=logger)

        self.binary_mask: Float[Tensor, "layers seq_len"] = torch.from_numpy(
            np.load("./datasets/other/elephant_binary.npy", allow_pickle=True)
        )

        num_layers, seq_len = self.binary_mask.shape
        assert num_layers == 32, "Only tested for 32 layers"

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to -inf where rep_loss_mask is False
        # obf_metrics = torch.where(rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf"))
        b_size, num_layers, seq_len = obf_metrics.shape

        # Apply self.binary_mask to obf_metrics
        mask: Float[Tensor, "layers-1 len"] = self.binary_mask.to(obf_metrics.device)
        # mask[mask == 0] = float(-1)

        mask_num_layers, mask_seq_len = mask.shape

        # Pad with 0s on the bottom of dim 1 to match num_layers
        if mask_num_layers < num_layers:
            padding = torch.zeros(
                num_layers - mask_num_layers, mask_seq_len, device=mask.device
            )
            mask = torch.cat([mask, padding], dim=0)
        else:
            mask = mask[:num_layers, :]

        mask = mask.to(torch.bool)

        # binary_masks = []
        # for loss_mask_row, obf_metric_row in zip(rep_loss_mask, obf_metrics):

        #    loss_mask_row: Float[Tensor, "seq_len"]
        #    obf_metric_row: Float[Tensor, "layers seq_len"]

        #    # loss_mask_row is all falses then all trues, find index of first true
        #    first_true_index = loss_mask_row.nonzero()[0]

        #    # create a copy of binary mask and left pad with 0s up to first_true_index
        #    left_pad = torch.zeros(num_layers, first_true_index, device=mask.device)
        #    mask_row = torch.cat([left_pad, mask.clone()], dim=1)
        #    new_mask_seq_len = mask_row.shape[1]

        #    if new_mask_seq_len < seq_len:
        #        padding = torch.zeros(num_layers, seq_len - new_mask_seq_len, device=mask.device)
        #        mask_row = torch.cat([mask_row, padding], dim=-1)
        #    else:
        #        mask_row = mask_row[:, :seq_len]

        #    new_mask_seq_len = mask_row.shape[1]
        #    assert (
        #        new_mask_seq_len == seq_len
        #    ), f"Expected mask_row to be of length {seq_len}, got {new_mask_seq_len}"

        #    binary_masks.append(mask_row)

        # Plot and save one of the masks
        # plt.figure(figsize=(5, 5))
        # plt.imshow(binary_masks[0].cpu(), cmap='binary')
        # plt.title('Binary Mask')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('./saved/images/binary_mask.png')
        # plt.close()

        # Apply binary masks to obf_metrics
        # Apply binary masks to obf_metrics
        # masked_obf_metrics = binary_mask * obf_metrics

        # output = masked_obf_metrics.max(dim=2).values.mean(dim=1)

        # binary_mask = torch.stack(binary_masks)
        mask = mask.unsqueeze(0).repeat(b_size, 1, 1)

        label = ~mask
        # Cut scores to length of mask
        assert seq_len >= mask_seq_len
        obf_metrics = obf_metrics[:, :, :mask_seq_len]
        # Expand label to match obf_metrics shape
        label = label.flip(0).float()
        label = label.flip(1)

        # Get MSE between obf_metrics and label, keeping batch dimension
        loss = (obf_metrics - label) ** 2
        output = loss.mean(dim=(1, 2))

        assert output.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {output.shape}"

        return output
