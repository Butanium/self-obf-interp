from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import (
    ForwardReturn,
    GCGParams,
    GenReturn,
    ModelBase,
    ModelConfig,
    SoftParams,
)
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


class Llama3SoftPrompted(HFSoftPrompted):

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "Llama" in self.model.__class__.__name__


class Llama3HardPrompted(HFHardPrompted):

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "Llama" in self.model.__class__.__name__
