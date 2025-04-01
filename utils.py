"""General utility functions for self-obfuscation experiments."""

import torch as th
import nnsight as nns
from nnterp.nnsight_utils import get_layer_output, get_num_layers, get_layer
from pathlib import Path


def keep_first_sequence(bool_tensor):
    # Get indices where value changes (1 -> 0 or 0 -> 1)
    changes = bool_tensor[:, 1:] != bool_tensor[:, :-1]

    # Prepend the first value to changes
    changes = th.cat([bool_tensor[:, :1], changes], dim=1)

    # Use cumsum to identify different sequences
    sequence_num = th.cumsum(changes, dim=1)

    # Keep only values from first sequence of 1s (sequence_num == 1)
    return bool_tensor & (sequence_num == 1)


def get_input_and_target_masks(
    assistant_mask: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    input_mask = keep_first_sequence(assistant_mask)
    target_mask = keep_first_sequence(assistant_mask & ~input_mask)
    return input_mask, target_mask


@th.no_grad()
def patch_pos(samples, pos_list, source_model, target_model, max_layer):
    """
    Patch token positions in the target model with activations from the source model.

    Args:
        samples: Input samples.
        pos_list: List of token positions (integers) to patch for each sample.
        source_model: Model to get activations from.
        target_model: Model to patch.
        max_layer: Maximum layer index.

    Returns:
        Tensor of activations from the patched model.
    """
    layer_acts = []
    print("collecting source model activations")
    with source_model.trace(
        dict(
            input_ids=samples.input_ids.to(source_model.device),
            attention_mask=samples.attention_mask.to(source_model.device),
        )
    ):
        for layer in range(max_layer + 1):
            acts = []
            for i, pos in enumerate(pos_list):
                assert (
                    0 <= pos < samples.input_ids.shape[1]
                ), f"pos: {pos} is out of bounds for input_ids: {samples.input_ids.shape[1]}"
                acts.append(get_layer_output(source_model, layer)[i, pos].save())
            layer_acts.append(acts)

    probe_acts = []
    print("patching target model")
    with target_model.trace(
        dict(input_ids=samples.input_ids, attention_mask=samples.attention_mask)
    ):
        for layer in range(max_layer + 1):
            for i, pos in enumerate(pos_list):
                assert (
                    0 <= pos < samples.input_ids.shape[1]
                ), f"pos: {pos} is out of bounds for input_ids: {samples.input_ids.shape[1]}"
                get_layer_output(target_model, layer)[i, pos] = layer_acts[layer][i]

        get_layer(target_model, max_layer).output.stop()
        probe_acts = get_layer_output(target_model, max_layer).cpu().save()

    print("patching finished")
    return probe_acts


def patching_exp(
    adv_samples,
    probe,
    source_model,
    target_model,
    tokenizer,
    max_layer,
    message_template_path,
):
    """
    Run a patching experiment.

    Args:
        adv_samples: Adversarial samples to use.
        probe: Probe to evaluate with.
        source_model: Model to get activations from.
        target_model: Model to patch.
        tokenizer: Tokenizer to use.
        max_layer: Maximum layer index.
        chat_template_path: Path to chat template file.

    Returns:
        Tensor of probe scores.
    """
    # Load chat template if provided
    with open(message_template_path, "r") as f:
        message_template = f.read()

    # Create conversations from samples
    convs = [
        [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
        for _, sample in adv_samples.iterrows()
    ]

    # Apply chat template
    convs = tokenizer.apply_chat_template(
        convs,
        padding=True,
        return_tensors="pt",
        return_dict=True,
        chat_template=message_template,
        return_assistant_tokens_mask=True,
    )
    messages_mask = convs.assistant_masks

    # Find token positions for the target word in each sample
    tok_pos = [
        get_first_word_tokens_pos(tokenizer, conv, sample["word"])
        for conv, (_, sample) in zip(convs.input_ids, adv_samples.iterrows())
    ]
    print(f"Token positions: {tok_pos}")

    # Validate dimensions and check for invalid positions
    assert (
        len(tok_pos) == len(adv_samples) == len(convs.input_ids)
    ), "Mismatched dimensions"

    # Patch token positions
    probe_acts = patch_pos(convs, tok_pos, source_model, target_model, max_layer)

    # Extract input and target masks
    input_mask, target_mask = get_input_and_target_masks(messages_mask)
    assert input_mask.shape == target_mask.shape, "Mismatched mask shapes"

    # Calculate input length
    input_len = input_mask.sum(dim=-1)
    print(f"input_mask: {input_mask}\nAssist mask: {messages_mask.tolist()}")
    assert (
        input_len == input_len[0]
    ).all(), f"Variable input lengths not supported, found: {input_len}"
    input_len = th.arange(input_mask.shape[1])[input_mask[0]].max()

    # Extract representations and calculate probe score
    input_reps = probe_acts[:, :input_len, :]
    start_target_reps = th.arange(target_mask.shape[1])[target_mask[0]].min()
    target_reps = probe_acts[:, start_target_reps:, :]

    probe_score = probe.predict_example(
        input_reps=input_reps,
        target_reps=target_reps,
        target_mask=target_mask[:, start_target_reps:],
    )

    print(f"Probe score: {probe_score}")
    return probe_score


def get_first_word_tokens_pos(tokenizer, prompt_or_tokens, word):
    """
    Find the last token position of a specific word in tokenized text.

    Args:
        tokenizer: The tokenizer to use.
        prompt_or_tokens: The text or tokenized text to search in.
        word: The word to find.

    Returns:
        Integer representing the position of the last token of the word.
    """
    if isinstance(prompt_or_tokens, str):
        tokens = tokenizer.tokenize(prompt_or_tokens)
    elif len(prompt_or_tokens) == 0:
        raise ValueError("Empty input")
    elif not isinstance(prompt_or_tokens[0], str):
        tokens = tokenizer.convert_ids_to_tokens(prompt_or_tokens)
    else:
        tokens = prompt_or_tokens

    tokens_queue = tokens.copy()
    curr_prompt = ""
    last_idx = 0
    while word not in curr_prompt:
        if len(tokens_queue) == 0:
            raise ValueError(f"Word {word} not found in prompt {prompt_or_tokens}")
        curr_prompt += tokens_queue.pop(0)
        last_idx += 1
    curr_end_prompt = ""
    for i in range(last_idx - 1, -1, -1):
        curr_end_prompt = tokens[i] + curr_end_prompt
        if word in curr_end_prompt:
            return i  # Return just the last position instead of a list

    raise ValueError(f"Error while finding word {word} in prompt {prompt_or_tokens}")


def layer_sweep(target_model, source_model, probe, examples, num_layers, max_layer):
    """
    Perform a layer sweep experiment by stitching different sets of layers.

    Args:
        target_model: Model to modify.
        source_model: Model to take layers from.
        probe: Probe to evaluate with.
        examples: Examples to evaluate on.
        num_layers: Number of consecutive layers to stitch.
        max_layer: Maximum layer index.

    Returns:
        List of probe scores for each starting layer.
    """
    from models import stitch_model, unstitch_model
    from probes import get_probe_score

    results = []
    all_layers = list(range(max_layer + 1))

    for i in range(max_layer + 1 - num_layers + 1):
        layers_to_stitch = all_layers[i : i + num_layers]
        stitch_model(target_model, source_model, layers_to_stitch)
        results.append(
            get_probe_score(probe, examples, model=target_model).mean().item()
        )
        unstitch_model(target_model, layers_to_stitch)

    return results
