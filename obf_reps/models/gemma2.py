from typing import List

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import GenReturn, ModelBase, ModelConfig
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


#   TODO: HF Gemma generate is currently broken, so
#       we are using this fix. It's a temporary fix,
#       as soon as HF updates their code, remove this.
#       PR to track is https://github.com/huggingface/transformers/pull/32932
def gemma_generate(
    model: ModelBase,
    input_ids: Int64[Tensor, "b_size input_seq_len"],
    input_attn_mask: Bool[Tensor, "b_size input_seq_len"],
    max_new_tokens: int = 20,
    use_tunable_params: bool = True,
    **generate_kwargs,
) -> GenReturn:
    """Generate text from input IDs using the model's generate function."""

    input_embeds, attention_mask = model._convert_ids_to_input_embeds(
        input_ids, input_attn_mask, use_tunable_params
    )

    batch_size: int = input_embeds.shape[0]
    input_no_embeds_seq_len: int = input_ids.shape[1]
    input_seq_len: int = input_embeds.shape[1]

    assert batch_size == 1, "Temporary gemma generate only supports batch size 1"

    gen_ids_list = []
    output = model.model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    for _ in range(max_new_tokens):
        logits = output.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        gen_ids_list.append(next_token_id)
        if next_token_id == model.tokenizer.eos_token_id:
            break
        next_token_embeds = model.model.get_input_embeddings()(next_token_id)
        input_embeds = torch.cat([input_embeds, next_token_embeds.unsqueeze(0)], dim=1)
        output = model.model(inputs_embeds=input_embeds, output_hidden_states=True)

    # When forward with input_embeds, model should only return ids for generation
    gen_ids: Int64[Tensor, "b_size gen_len"] = torch.stack(gen_ids_list, dim=1)

    # Get text and reps
    gen_text: List[str] = model.tokenizer.batch_decode(
        gen_ids, skip_special_tokens=False
    )
    input_text: List[str] = model.tokenizer.batch_decode(
        input_ids, skip_special_tokens=False
    )

    hidden_states: Float[Tensor, "layers b_size seq_len hidden_size"] = torch.stack(
        output.hidden_states
    )
    hidden_states: Float[Tensor, "b_size layers seq_len hidden_size"] = rearrange(
        hidden_states,
        "layers b_size seq_len hidden_size -> b_size layers seq_len hidden_size",
    )

    # TODO: This includes reps for the tunable params
    input_reps: Float[Tensor, "b_size layers input_seq_len hidden_size"] = (
        hidden_states[:, :, :input_seq_len, :]
    )

    gen_reps: Float[Tensor, "b_size layers gen_seq_len hidden_size"] = hidden_states[
        :, :, input_seq_len:, :
    ]

    return GenReturn(
        input_text=input_text,
        gen_text=gen_text,
        input_ids=input_ids,
        gen_ids=gen_ids,
        input_reps=input_reps,
        gen_reps=gen_reps,
    )


class Gemma2bSoftPrompted(HFSoftPrompted):

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "Gemma2ForCausalLM" == self.model.__class__.__name__


class Gemma2bHardPrompted(HFHardPrompted):

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "Gemma2ForCausalLM" == self.model.__class__.__name__
