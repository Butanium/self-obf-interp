from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import ModelConfig
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


class LlamaRRSoftPrompted(HFSoftPrompted):

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "RR" in self.model.config._name_or_path


class LlamaRRHardPrompted(HFHardPrompted):

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "RR" in self.model.config._name_or_path
