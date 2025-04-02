"""General utility functions for self-obfuscation experiments."""

import torch as th
from nnterp.nnsight_utils import get_layer_output, get_num_layers, get_layer
from tqdm.auto import tqdm
from models import stitch_model, unstitch_model, get_model_reps

def keep_first_sequence(bool_tensor):
    assert (
        bool_tensor.dtype == th.bool
    ), f"Input tensor must be boolean, got {bool_tensor.dtype}"
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
    pos_tensor = th.tensor(pos_list, device=samples.input_ids.device)
    arange = th.arange(samples.input_ids.shape[0], device=samples.input_ids.device)
    layer_acts = []
    with source_model.trace(
        dict(
            input_ids=samples.input_ids.to(source_model.device),
            attention_mask=samples.attention_mask.to(source_model.device),
        )
    ):
        for layer in range(max_layer + 1):
            layer_acts.append(
                get_layer_output(source_model, layer)[arange, pos_tensor].save()
            )

    with target_model.trace(
        dict(input_ids=samples.input_ids, attention_mask=samples.attention_mask)
    ):
        for layer in range(max_layer + 1):
            get_layer_output(target_model, layer)[arange, pos_tensor] = layer_acts[
                layer
            ]

        probe_acts = get_layer_output(target_model, max_layer).save()
        get_layer(target_model, max_layer).output.stop()

    return probe_acts


def patching_exp(
    samples_dict,
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
        samples_dict: Dictionary of samples to use.
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
    result = {}
    for name, samples in tqdm(samples_dict.items(), desc="Patching experiments"):
        # Create conversations from samples
        convs = [
            [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["response"]},
            ]
            for _, sample in samples.iterrows()
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
        messages_mask = convs.assistant_masks.bool()

        # Find token positions for the target word in each sample
        tok_pos = [
            get_first_word_token_pos(tokenizer, conv, sample["word"])
            for conv, (_, sample) in zip(convs.input_ids, samples.iterrows())
        ]
        assert (
            len(tok_pos) == len(samples) == len(convs.input_ids)
        ), "Mismatched dimensions"

        # Patch token positions
        probe_acts = patch_pos(convs, tok_pos, source_model, target_model, max_layer)

        # Extract input and target masks
        input_mask, target_mask = get_input_and_target_masks(messages_mask)
        assert input_mask.shape == target_mask.shape, "Mismatched mask shapes"
        start_target_reps = th.arange(target_mask.shape[1])[target_mask[0]].min()
        target_reps = probe_acts[:, start_target_reps:, :].unsqueeze(1)  # add layer dim
        target_mask = target_mask[:, start_target_reps:].to(target_reps.device)
        probe_score = probe.predict_example(
            input_reps=None,
            target_reps=target_reps,
            target_mask=target_mask,
        )

        result[name] = probe_score.cpu().numpy()

    return result


def get_first_word_token_pos(tokenizer, prompt_or_tokens, word):
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


def run_stitching_experiment(
    target_model, source_model, get_results_fn, stop_at_layer=None
):

    if stop_at_layer is None:
        stop_at_layer = get_num_layers(target_model.model) - 1

    results = []
    unstitch_model(target_model)
    for i in tqdm(range((stop_at_layer + 1)), desc="Running stitching experiment"):
        stitch_model(target_model, source_model, list(range(i)))
        results.append(get_results_fn(target_model))
        unstitch_model(target_model, list(range(i)))

    return results



@th.no_grad()
def get_dist_between_reps(entries, base_model, obf_model):
    """Calculate distance metrics between base and obfuscated model representations.

    Args:
        entries: Examples to evaluate on.
        base_model: The base model.
        obf_model: The obfuscated model.

    Returns:
        Tuple of various distance metrics.
    """
    layers = list(range(len(base_model.model.model.layers)))
    base_reps = get_model_reps(base_model, entries, layers=layers)
    obf_reps = get_model_reps(obf_model, entries, layers=layers)

    # Input representation norms
    input_base_reps_norms = (
        th.norm(base_reps.input_reps, dim=-1)
        .transpose(0, 1)
        .reshape(len(layers), -1)
        .mean(dim=-1)
    )
    input_obf_reps_norms = (
        th.norm(obf_reps.input_reps, dim=-1)
        .transpose(0, 1)
        .reshape(len(layers), -1)
        .mean(dim=-1)
    )
    input_reps_norms = (
        th.norm(base_reps.input_reps - obf_reps.input_reps, dim=-1)
        .transpose(0, 1)
        .reshape(len(layers), -1)
        .mean(dim=-1)
    )
    input_reps_norms_normalized = 2 * (
        th.norm(base_reps.input_reps - obf_reps.input_reps, dim=-1)
        / (th.norm(base_reps.input_reps, dim=-1) + th.norm(obf_reps.input_reps, dim=-1))
    ).transpose(0, 1).reshape(len(layers), -1).mean(dim=-1)

    # Target representation norms
    target_base_reps_norms = (
        th.norm(base_reps.target_reps, dim=-1)
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    target_obf_reps_norms = (
        th.norm(obf_reps.target_reps, dim=-1)
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    target_reps_norms = (
        th.norm(base_reps.target_reps - obf_reps.target_reps, dim=-1)
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    target_reps_norms_normalized = (
        (
            2
            * th.norm(base_reps.target_reps - obf_reps.target_reps, dim=-1)
            / (
                th.norm(base_reps.target_reps, dim=-1)
                + th.norm(obf_reps.target_reps, dim=-1)
            )
        )
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )

    # KL divergence metrics
    kl_div_input = th.nn.functional.kl_div(
        th.log_softmax(obf_reps.input_logits, dim=-1),
        th.log_softmax(base_reps.input_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    )
    kl_div_target = (
        th.nn.functional.kl_div(
            th.log_softmax(obf_reps.target_logits[obf_reps.loss_mask], dim=-1),
            th.log_softmax(base_reps.target_logits[base_reps.loss_mask], dim=-1),
            log_target=True,
            reduction="sum",
        )
        / base_reps.loss_mask.sum()
    )

    # Statistical distance metrics
    base_probs_input = th.softmax(base_reps.input_logits, dim=-1)
    obf_probs_input = th.softmax(obf_reps.input_logits, dim=-1)
    base_probs_target = th.softmax(base_reps.target_logits, dim=-1)
    obf_probs_target = th.softmax(obf_reps.target_logits, dim=-1)
    stat_dist_input = (base_probs_input - obf_probs_input).abs().sum(dim=-1).mean()
    stat_dist_target = (base_probs_target - obf_probs_target).abs().sum(dim=-1).mean()

    return (
        input_reps_norms.cpu().numpy(),
        target_reps_norms.cpu().numpy(),
        input_reps_norms_normalized.cpu().numpy(),
        target_reps_norms_normalized.cpu().numpy(),
        input_base_reps_norms.cpu().numpy(),
        input_obf_reps_norms.cpu().numpy(),
        target_base_reps_norms.cpu().numpy(),
        target_obf_reps_norms.cpu().numpy(),
        kl_div_input.item(),
        kl_div_target.item(),
        stat_dist_input.item(),
        stat_dist_target.item(),
    )
