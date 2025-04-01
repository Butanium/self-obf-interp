# %%
try:
    from IPython import get_ipython

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except NameError:
    # Not running in IPython/Jupyter environment
    print("Not running in IPython/Jupyter environment")

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import json
import pickle
from obf_reps.metrics import TrainableMetric
from copy import deepcopy
import os
import torch
import torch as th

from collections import defaultdict
import pickle
from jaxtyping import Bool, Float, Int64
from obf_reps.logging import Logger
from obf_reps.metrics import MLPMetric, TrainableMetric
from obf_reps.models.hf import HFHardPrompted, HFHardPromptedWithSelectableLayers
from obf_reps.models import ModelConfig
from pathlib import Path
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

import os
import torch


WORK_DIR = Path(__file__).parent


def load_word_to_probe_dict(probe_checkpoint_dir: str) -> dict[str, TrainableMetric]:
    """
    Load a dictionary mapping words to their corresponding probe objects from a given directory.

    The function assumes that each probe is saved as a .pkl file in the specified directory,
    where the filename (without the .pkl extension) is the word. For example, a file named
    "bomb.pkl" will be loaded as the probe for the word "bomb".

    Args:
        probe_checkpoint_dir (str): Path to the directory containing the probe checkpoint files.

    Returns:
        dict[str, TrainableMetric]: A dictionary mapping word (str) to the loaded probe object.
    """
    if not os.path.isdir(probe_checkpoint_dir):
        raise ValueError(f"Provided path is not a directory: {probe_checkpoint_dir}")

    word_to_probe: dict[str, TrainableMetric] = {}
    # Iterate over all files in the directory
    for filename in os.listdir(probe_checkpoint_dir):
        if filename.endswith(".pkl"):
            word = filename[:-4]  # Remove the ".pkl" extension to get the word
            word = word.replace("train_", "").replace("test_", "")
            file_path = os.path.join(probe_checkpoint_dir, filename)
            with open(file_path, "rb") as f:
                probe = pickle.load(f)
            if not isinstance(probe, TrainableMetric):
                raise ValueError(f"Loaded object is not a TrainableMetric: {file_path}")
            word_to_probe[word] = probe

    return word_to_probe


def load_local_model(
    checkpoint_path: Optional[str] = None,
    model_name: str = "google/gemma-2-9b-it",
    device=None,
):
    """Load a HF model, optionally from a checkpoint.

    Args:
        checkpoint_path: Optional path to a model checkpoint. If None, loads base model.
    """
    print("Loading model...")
    if device is None:
        model_config = ModelConfig()
    else:
        model_config = ModelConfig(device=device)

    if checkpoint_path is not None:
        # Load model weights from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=model_config.device,
            torch_dtype=model_config.model_dtype,
        )
        model.eval()
        if not model_config.requires_grad:
            for p in model.parameters():
                p.requires_grad = False

        # Load and configure tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token:
            pass
        elif tokenizer.unk_token:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Create the wrapped model
        model = HFHardPromptedWithSelectableLayers(
            model=model, tokenizer=tokenizer, config=model_config
        )
    else:
        # Load everything from base model
        model = HFHardPromptedWithSelectableLayers.load_model(model_name, model_config)

    return model


# %%
test_words = [
    "gun",
    "apple",
    "elephant",
    "SQL",
    "cloud",
    "jester",
    "mathematical",
    "samantha",
]  # Could also just get this from probe names
probes_checkpoint_path = (
    WORK_DIR
    / "probes_20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_finetuned"
)
model_checkpoint_path = (
    WORK_DIR
    / "20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_good"
)
test_data_path = (
    WORK_DIR / "data/conditional_test_data_gemma_2_9b_instruct_20250226_010745.json"
)
USE_RESPONSE_TOKEN_IDS = True  # Cautious preservation of special tokens in response....
layers_to_probe = [12]  # Could also just get this from metadata

# STEP 1: Load stuff
word_to_probe = load_word_to_probe_dict(probes_checkpoint_path)
self_obf_model = load_local_model(checkpoint_path=model_checkpoint_path)
cpu_obf_model = load_local_model(checkpoint_path=model_checkpoint_path, device="cpu")
base_model = load_local_model()
with open(test_data_path, "r") as f:
    test_data = json.load(f)["data"]
test_data_df = pd.DataFrame(test_data)

# %%
gun_probe = word_to_probe["gun"]
# Create base examples for each word
bomb_examples = test_data_df.query("word == 'bomb'")
gun_examples = test_data_df.query("word == 'gun'")
elephant_examples = test_data_df.query("word == 'elephant'")

# Define the words we want to create variations for
words = ["bomb", "gun", "elephant"]
entries = {}

# Generate all combinations of word and adversarial word
for word in words:
    base_examples = test_data_df.query(f"word == '{word}'")
    vanilla_examples = base_examples.query("adversarial_word.isna()")
    entries[f"{word.capitalize()} vanilla"] = vanilla_examples
    # Same as word adv word actually
    # hybrid_examples = vanilla_examples.copy()
    # hybrid_examples["prompt"] = hybrid_examples["prompt"].apply(
    #     lambda x: f"You are being probed for '{word}'. {x}"
    # )
    # entries[f"{word.capitalize()} hybrid adv {word}"] = hybrid_examples

    for adv_word in words:
        if word == adv_word:
            # For cases where the word is the same as the adversarial word
            entries[f"{word.capitalize()} adv {adv_word}"] = base_examples.query(
                f"adversarial_word == '{adv_word}'"
            )
        else:
            # For cases where we need to replace the adversarial word in the prompt
            # Find examples with any adversarial word (doesn't matter which one)
            source_examples = base_examples.query(f"adversarial_word == '{word}'")
            # Copy the examples and replace the adversarial word in the prompt
            modified_examples = source_examples.copy()
            modified_examples["prompt"] = modified_examples["prompt"].apply(
                lambda x: x.replace(
                    f"You are being probed for '{word}'",
                    f"You are being probed for '{adv_word}'",
                )
            )
            entries[f"{word.capitalize()} adv {adv_word}"] = modified_examples


# %%
@th.no_grad()
def get_model_reps(model, entry, layers=[12]):
    return model.forward_from_string(
        input_text=entry["prompt"],
        target_text=list(entry["response"]),
        add_chat_template=True,
        use_tunable_params=False,
        layers_to_probe=layers,
    )


@th.no_grad()
def get_probe_score(probe, entry, model=self_obf_model):
    forward_return = get_model_reps(model, entry)
    return probe.predict_example(
        input_reps=forward_return.input_reps,
        target_reps=forward_return.target_reps,
        target_mask=forward_return.loss_mask,
    )


# %%
for entry_name, entry in entries.items():
    print(
        f"{entry_name}:\nSelf Obf model:{get_probe_score(gun_probe, entry).mean().item()}"
    )
    print(
        f"Base model: {get_probe_score(gun_probe, entry, model=base_model).mean().item()}"
    )
    print("-" * 100)


# %%
gun_adv_gun_examples = entries["Gun adv gun"]


def stitch_model(target_model, source_model, layers):
    if not hasattr(target_model, "original_layers"):
        target_model.original_layers = [
            layer for layer in target_model.model.model.layers
        ]
    for layer in layers:
        target_model.model.model.layers[layer] = source_model.model.model.layers[layer]
    return target_model


def unstitch_model(target_model, layers):
    if layers is None:
        layers = list(range(len(target_model.original_layers)))
    for layer in layers:
        target_model.model.model.layers[layer] = target_model.original_layers[layer]
    return target_model


# %%
res_adv_gun = []
res_vanilla_gun = []
for i in range(layers_to_probe[0] + 1):
    stitch_model(self_obf_model, base_model, [i])
    adv_score = (
        get_probe_score(gun_probe, gun_adv_gun_examples, model=self_obf_model)
        .mean()
        .item()
    )
    vanilla_score = (
        get_probe_score(gun_probe, entries["Gun vanilla"], model=self_obf_model)
        .mean()
        .item()
    )
    res_adv_gun.append(adv_score)
    res_vanilla_gun.append(vanilla_score)
    print(f"Layer 0-{i}:\nAdv gun probe score: {adv_score}")
    print(f"Vanilla gun probe score: {vanilla_score}")
    print("-" * 100)
unstitch_model(self_obf_model, None)
res_adv_gun_rev = []
res_vanilla_gun_rev = []
for i in range(layers_to_probe[0], -1, -1):
    stitch_model(self_obf_model, base_model, [i])
    adv_score = (
        get_probe_score(gun_probe, gun_adv_gun_examples, model=self_obf_model)
        .mean()
        .item()
    )
    vanilla_score = (
        get_probe_score(gun_probe, entries["Gun vanilla"], model=self_obf_model)
        .mean()
        .item()
    )
    res_adv_gun_rev.append(adv_score)
    res_vanilla_gun_rev.append(vanilla_score)
    print(
        f"Layer {i}-n\nAdv gun probe score: {adv_score}\nVanilla gun probe score: {vanilla_score}"
    )
    print("-" * 100)
_ = unstitch_model(self_obf_model, None)
# %%
stitch_model(self_obf_model, base_model, [5, 6])
layers_5_6_adv_gun = (
    get_probe_score(gun_probe, gun_adv_gun_examples, model=self_obf_model).mean().item()
)
layers_5_6_vanilla_gun = (
    get_probe_score(gun_probe, entries["Gun vanilla"], model=self_obf_model)
    .mean()
    .item()
)
_ = unstitch_model(self_obf_model, None)


# %%
def layer_sweep(target_model, source_model, probe, examples, num_layers):
    results = []
    all_layers = list(range(layers_to_probe[0] + 1))
    for i in range(layers_to_probe[0] + 1):
        stitch_model(target_model, source_model, all_layers[i : i + num_layers])
        results.append(
            get_probe_score(probe, examples, model=target_model).mean().item()
        )
        unstitch_model(target_model, all_layers[i : i + num_layers])
    return results


layer_sweep_results = {}
for num_layers in [1, 3, 5]:
    layer_sweep_results[f"{num_layers}_layer_from_obf_adv_gun"] = layer_sweep(
        base_model, self_obf_model, gun_probe, entries["Gun adv gun"], num_layers
    )
    layer_sweep_results[f"{num_layers}_layer_from_base_adv_gun"] = layer_sweep(
        self_obf_model, base_model, gun_probe, entries["Gun adv gun"], num_layers
    )
    layer_sweep_results[f"{num_layers}_layer_from_obf_vanilla_gun"] = layer_sweep(
        base_model, self_obf_model, gun_probe, entries["Gun vanilla"], num_layers
    )
    layer_sweep_results[f"{num_layers}_layer_from_base_vanilla_gun"] = layer_sweep(
        self_obf_model, base_model, gun_probe, entries["Gun vanilla"], num_layers
    )


# %%
import plotly.graph_objects as go
from pathlib import Path

fig = make_subplots(
    rows=1,
    cols=2,
    x_title="Layer",
    subplot_titles=["Patch Obfuscated → Base", "Patch Base → Obfuscated"],
)

# Plot single layer sweep results
# Plot layer sweep results using a loop
colors = {"adv_gun": "green", "vanilla_gun": "purple"}
layer_types = {1: "Single Layer", 3: "Three Layers", 5: "Five Layers"}
dash_styles = {1: None, 3: "dash", 5: "longdashdot"}
marker_styles = {1: "circle", 3: "square", 5: "diamond"}
for col, from_model in enumerate(["obf", "base"]):
    for num_layers in [1, 3, 5]:
        for example_type in ["adv_gun", "vanilla_gun"]:
            data_key = f"{num_layers}_layer_from_{from_model}_{example_type}"
            fig.add_trace(
                go.Scatter(
                    x=list(range(layers_to_probe[0] + 1)),
                    y=layer_sweep_results[data_key],
                    mode="lines+markers",
                    name=f"{num_layers} layers - {'Adversarial' if example_type == 'adv_gun' else 'Vanilla'} Gun Score",
                    line=dict(color=colors[example_type], dash=dash_styles[num_layers]),
                    showlegend=col == 0,
                    marker=dict(symbol=marker_styles[num_layers]),
                ),
                row=1,
                col=col + 1,
            )


# Update layout
fig.update_layout(
    title="Layer Sweep Results",
    xaxis_title="Layer",
    yaxis_title="Probe Score",
    legend_title="Metrics",
    width=800,
)
fig.write_html("results/layer_sweep_results.html")
# Save the figure as a PNG image
fig.write_image("results/layer_sweep_results.png", width=1200, height=800, scale=2)

fig.show()
# %%

# Create a single figure
fig = go.Figure()

# Forward stitching traces
fig.add_trace(
    go.Scatter(
        x=list(range(layers_to_probe[0] + 1)),
        y=res_adv_gun,
        mode="lines+markers",
        name="Forward - Adversarial Gun Score",
        line=dict(color="red"),
    )
)

fig.add_trace(
    go.Scatter(
        x=list(range(layers_to_probe[0] + 1)),
        y=res_vanilla_gun,
        mode="lines+markers",
        name="Forward - Vanilla Gun Score",
        line=dict(color="blue"),
    )
)

# Reverse stitching traces
fig.add_trace(
    go.Scatter(
        x=list(range(layers_to_probe[0], -1, -1)),
        y=res_adv_gun_rev,
        mode="lines+markers",
        name="Reverse - Adversarial Gun Score",
        line=dict(color="red", dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=list(range(layers_to_probe[0], -1, -1)),
        y=res_vanilla_gun_rev,
        mode="lines+markers",
        name="Reverse - Vanilla Gun Score",
        line=dict(color="blue", dash="dash"),
    )
)

# Add horizontal lines for layer 5-6 intervention
fig.add_shape(
    type="line",
    x0=0,
    y0=layers_5_6_adv_gun,
    x1=layers_to_probe[0],
    y1=layers_5_6_adv_gun,
    line=dict(
        color="red",
        width=2,
        dash="dot",
    ),
    name="Adversarial Gun Score (Layer 5-6)",
)

fig.add_shape(
    type="line",
    x0=0,
    y0=layers_5_6_vanilla_gun,
    x1=layers_to_probe[0],
    y1=layers_5_6_vanilla_gun,
    line=dict(
        color="blue",
        width=2,
        dash="dot",
    ),
    name="Vanilla Gun Score (Layer 5-6)",
)

# Add annotations for the intervention lines
fig.add_annotation(
    x=layers_to_probe[0],
    y=layers_5_6_adv_gun,
    text="Layer 5-6 Adversarial",
    showarrow=True,
    arrowhead=1,
    ax=-40,
    ay=0,
    font=dict(color="red", size=12),
)

fig.add_annotation(
    x=layers_to_probe[0],
    y=layers_5_6_vanilla_gun,
    text="Layer 5-6 Vanilla",
    showarrow=True,
    arrowhead=1,
    ax=-40,
    ay=0,
    font=dict(color="blue", size=12),
)


# Update layout
fig.update_layout(
    title="Gun Probe Scores with Layer Stitching",
    xaxis_title="Layer",
    yaxis_title="Probe Score",
    height=600,
    width=1200,
    # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


Path("results").mkdir(exist_ok=True)
fig.write_html("results/gun_probe_scores_with_layer_stitching.html")
fig.write_image(
    "results/gun_probe_scores_with_layer_stitching.png", width=1200, height=800, scale=2
)
fig.show()


# %%
@th.no_grad()
def get_dist_between_reps(entries):
    layers = list(range(len(base_model.model.model.layers)))
    base_reps = get_model_reps(base_model, entries, layers=layers)
    obf_reps = get_model_reps(self_obf_model, entries, layers=layers)

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


(
    input_reps_norms_adv_gun,
    target_reps_norms_adv_gun,
    input_reps_norms_normalized_adv_gun,
    target_reps_norms_normalized_adv_gun,
    input_base_reps_norms_adv_gun,
    input_obf_reps_norms_adv_gun,
    target_base_reps_norms_adv_gun,
    target_obf_reps_norms_adv_gun,
    kl_div_input_adv_gun,
    kl_div_target_adv_gun,
    stat_dist_input_adv_gun,
    stat_dist_target_adv_gun,
) = get_dist_between_reps(entries["Gun adv gun"])
(
    input_reps_norms_vanilla_gun,
    target_reps_norms_vanilla_gun,
    input_reps_norms_normalized_vanilla_gun,
    target_reps_norms_normalized_vanilla_gun,
    input_base_reps_norms_vanilla_gun,
    input_obf_reps_norms_vanilla_gun,
    target_base_reps_norms_vanilla_gun,
    target_obf_reps_norms_vanilla_gun,
    kl_div_input_vanilla_gun,
    kl_div_target_vanilla_gun,
    stat_dist_input_vanilla_gun,
    stat_dist_target_vanilla_gun,
) = get_dist_between_reps(entries["Gun vanilla"])
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

# Left subplot - Raw values
ax1.plot(
    input_reps_norms_adv_gun,
    label="Input reps (adversarial gun)",
    marker="o",
    color="red",
    linewidth=2,
    linestyle="-",
)
ax1.plot(
    input_reps_norms_vanilla_gun,
    label="Input reps (vanilla gun)",
    marker="x",
    color="blue",
    linewidth=2,
    linestyle="-",
)
ax1.plot(
    target_reps_norms_adv_gun,
    label="Target reps (adversarial gun)",
    marker="s",
    color="darkred",
    linewidth=2,
    linestyle="--",
)
ax1.plot(
    target_reps_norms_vanilla_gun,
    label="Target reps (vanilla gun)",
    marker="+",
    color="darkblue",
    linewidth=2,
    linestyle="--",
)
ax1.set_title("Raw Representation Difference Norms", fontsize=14)
ax1.set_xlabel("Layer Index", fontsize=12)
ax1.set_ylabel("Norm of the Difference", fontsize=12)

# Middle subplot - Normalized values
ax2.plot(
    input_reps_norms_normalized_adv_gun,
    label="Input reps (adversarial gun)",
    marker="o",
    color="red",
    linewidth=2,
    linestyle="-",
)
ax2.plot(
    input_reps_norms_normalized_vanilla_gun,
    label="Input reps (vanilla gun)",
    marker="x",
    color="blue",
    linewidth=2,
    linestyle="-",
)
ax2.plot(
    target_reps_norms_normalized_adv_gun,
    label="Target reps (adversarial gun)",
    marker="s",
    color="darkred",
    linewidth=2,
    linestyle="--",
)
ax2.plot(
    target_reps_norms_normalized_vanilla_gun,
    label="Target reps (vanilla gun)",
    marker="+",
    color="darkblue",
    linewidth=2,
    linestyle="--",
)
ax2.set_title("Normalized Representation Difference Norms", fontsize=14)
ax2.set_xlabel("Layer Index", fontsize=12)
ax2.set_ylabel("Normalized Norm of the Difference", fontsize=12)

# Right subplot - Raw norms from both models
ax3.plot(
    input_base_reps_norms_adv_gun,
    label="Base model input (adversarial gun)",
    marker="o",
    color="red",
    linewidth=2,
    linestyle="-",
)
ax3.plot(
    input_base_reps_norms_vanilla_gun,
    label="Base model input (vanilla gun)",
    marker="x",
    color="blue",
    linewidth=2,
    linestyle="-",
)
ax3.plot(
    input_obf_reps_norms_adv_gun,
    label="Obf model input (adversarial gun)",
    marker="s",
    color="darkred",
    linewidth=2,
    linestyle="--",
)
ax3.plot(
    input_obf_reps_norms_vanilla_gun,
    label="Obf model input (vanilla gun)",
    marker="+",
    color="darkblue",
    linewidth=2,
    linestyle="--",
)
ax3.set_title("Raw Representation Norms", fontsize=14)
ax3.set_xlabel("Layer Index", fontsize=12)
ax3.set_ylabel("Norm of Representations", fontsize=12)

fig.suptitle("Representation Analysis: Base vs Self-Obfuscated Model", fontsize=16)

# Add grid and legend to all subplots
for ax in [ax1, ax2, ax3]:
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=9, loc="best")

# Improve overall appearance
plt.tight_layout()
plt.savefig("results/representation_norms_analysis.png", dpi=300)
plt.show()
# %%

# Create a single bar plot for both input and target KL divergence
fig = make_subplots(rows=1, cols=2)

# Add bars for input KL divergence
fig.add_trace(
    go.Bar(
        x=["Gun adv gun", "Gun vanilla"],
        y=[kl_div_input_adv_gun, kl_div_input_vanilla_gun],
        name="Input KL Divergence",
    ),
    row=1,
    col=1,
)

# Add bars for target KL divergence
fig.add_trace(
    go.Bar(
        x=["Gun adv gun", "Gun vanilla"],
        y=[kl_div_target_adv_gun, kl_div_target_vanilla_gun],
        name="Target KL Divergence",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=["Gun adv gun", "Gun vanilla"],
        y=[stat_dist_input_adv_gun, stat_dist_input_vanilla_gun],
        name="Input Stat Dist",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Bar(
        x=["Gun adv gun", "Gun vanilla"],
        y=[stat_dist_target_adv_gun, stat_dist_target_vanilla_gun],
        name="Target Stat Dist",
    ),
    row=1,
    col=2,
)

# Update layout
fig.update_layout(
    title_text="KL Divergence between Base Model and Self-Obfuscated Model",
    xaxis_title="Example Type",
    yaxis_title="KL Divergence",
    height=500,
    width=800,
    legend_title="Metric",
    barmode="group",  # Group bars side by side
)
fig.write_html("results/kl_divergence_between_base_and_self_obfuscated_model.html")
fig.write_image(
    "results/kl_divergence_between_base_and_self_obfuscated_model.png",
    width=1200,
    height=800,
    scale=2,
)
fig.show()
# %%
sample_entry = entries["Gun adv gun"].sample(1)
obf_reps = get_model_reps(
    self_obf_model,
    sample_entry,
    layers=list(range(len(self_obf_model.model.model.layers))),
)
base_reps = get_model_reps(
    base_model, sample_entry, layers=list(range(len(base_model.model.model.layers)))
)


# %%
def unembed_probs(reps, model):
    assert reps.dim() <= 3
    unembed = model.model.lm_head(reps)
    return th.softmax(unembed, dim=-1)


obf_probs = unembed_probs(obf_reps.input_reps[0], self_obf_model)
base_probs = unembed_probs(base_reps.input_reps[0], base_model)

# %%
from utils import plot_top_tokens

tokens = base_model.tokenizer.convert_ids_to_tokens(base_reps.input_ids[0].tolist())
tokens = list(filter(lambda x: x != "<|optim-location|>", tokens))
# %%
fig_obf = plot_top_tokens(
    obf_probs.cpu(),
    tokens,
    self_obf_model.tokenizer,
    height=800,
    width=1250,
    title="Self-Obfuscated Model",
    annot_size=10,
)
fig_base = plot_top_tokens(
    base_probs.cpu(),
    tokens,
    self_obf_model.tokenizer,
    height=800,
    width=1250,
    title="Base Model",
    annot_size=10,
)
fig_obf.write_html("results/top_tokens_obf.html")
fig_base.write_html("results/top_tokens_base.html")
fig_obf.write_image("results/top_tokens_obf.png", width=1200, height=800, scale=2)
fig_base.write_image("results/top_tokens_base.png", width=1200, height=800, scale=2)
# %%
# %%

# %%
reps = get_model_reps(
    self_obf_model, entries["Gun adv gun"], layers=list(range(layers_to_probe[0] + 1))
)
# %%
reps.input_reps.shape
# %%
reps.target_reps.shape
# %%
reps.loss_mask.sum(dim=-1)


# %%
# todo: patch the tokens of the object to avoid with base model
# todo: patch only mlps / attention head
def get_first_word_tokens_pos(tokenizer, prompt_or_tokens, word):
    if isinstance(prompt_or_tokens, str):
        tokens = tokenizer.tokenize(prompt_or_tokens)
    elif len(prompt_or_tokens) == 0:
        return []
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
    word_token_pos = []
    for i in range(last_idx - 1, -1, -1):
        curr_end_prompt = tokens[i] + curr_end_prompt
        word_token_pos.append(i)
        if word in curr_end_prompt:
            break
    else:
        raise ValueError(
            f"Error while finding word {word} in prompt {prompt_or_tokens}"
        )
    return word_token_pos


test_prompts = [
    "The apple is red and the banana is yellow.",
    "I love bananas and apples.",
    "Hello 'banana'",
    "bananabannaa",
]

for prompt in test_prompts:

    pos = get_first_word_tokens_pos(base_model.tokenizer, prompt, "banana")
    print(pos)
    start = False
    end = False
    out = ""
    for i, token in enumerate(tokens):
        if i in pos:
            if not start:
                start = True
                out += f"**"
        else:
            if start and not end:
                end = True
                out += f"**"
        out += token + "|"
    print(out)
# %%
from nnsight import LanguageModel

nns_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
nns_tokenizer.padding_side = "right"
nns_self_obf_model = LanguageModel(self_obf_model.model, tokenizer=nns_tokenizer)
nns_base_model = LanguageModel(base_model.model, tokenizer=nns_tokenizer)

# %%
test_conv = [
    {"role": "user", "content": "The apple is red and the banana is yellow."},
    {"role": "assistant", "content": "The apple is red and the banana is yellow."},
]
print(test_conv)
test_conv = base_model.tokenizer.apply_chat_template(
    test_conv, tokenize=True, add_special_tokens=False
)
print(test_conv)
test_tok_pos = get_first_word_tokens_pos(base_model.tokenizer, test_conv, "banana")
print(test_tok_pos)


# %%
from nnterp.nnsight_utils import get_layer_output, get_num_layers, get_layer
import nnsight as nns


@th.no_grad()
def patch_pos(samples, pos_list, source_model, target_model):
    layer_acts = []
    print("collecting source model activations")
    with source_model.trace(
        dict(
            input_ids=samples.input_ids.to(source_model.device),
            attention_mask=samples.attention_mask.to(source_model.device),
        )
    ):
        for layer in range(layers_to_probe[0] + 1):
            acts = []
            for i, pos in enumerate(pos_list):
                assert (
                    max(pos) < samples.input_ids.shape[1]
                ), f"pos: {pos} is out of bounds for input_ids: {samples.input_ids.shape[1]}"
                acts.append(get_layer_output(source_model, layer)[i][pos].save())
            layer_acts.append(acts)
    probe_acts = []
    print("patching target model")
    with target_model.trace(
        dict(input_ids=samples.input_ids, attention_mask=samples.attention_mask)
    ):
        for layer in range(layers_to_probe[0] + 1):
            for i, pos in enumerate(pos_list):
                assert (
                    max(pos) < samples.input_ids.shape[1]
                ), f"pos: {pos} is out of bounds for input_ids: {samples.input_ids.shape[1]}"
                nns.log("shapes:")
                nns.log(get_layer_output(target_model, layer).shape)
                nns.log(layer_acts[layer][i][pos].shape)
                nns.log("end shapes")
                get_layer_output(target_model, layer)[i][pos] = layer_acts[layer][i][
                    pos
                ]
        get_layer(target_model, layers_to_probe[0]).output.stop()
        probe_acts = get_layer_output(target_model, layers_to_probe[0]).cpu().save()
    print("patched finish")
    return probe_acts


with open("data/gemma_chat_template.jinja", "r") as f:
    chat_template = f.read()
from utils import get_input_and_target_masks


def patching_exp(adv_samples, probe, source_model, target_model):
    convs = [
        [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
        for _, sample in adv_samples.iterrows()
    ]
    convs = base_model.tokenizer.apply_chat_template(
        convs,
        padding=True,
        return_tensors="pt",
        return_dict=True,
        chat_template=chat_template,
        return_assistant_tokens_mask=True,
    )

    tok_pos = [
        get_first_word_tokens_pos(base_model.tokenizer, conv, sample["word"])
        for conv, (_, sample) in zip(convs.input_ids, adv_samples.iterrows())
    ]
    print(f"tok_pos: {tok_pos}")
    assert len(tok_pos) == len(adv_samples) == len(convs.input_ids)
    probe_acts = patch_pos(convs, tok_pos, source_model, target_model)
    input_mask, target_mask = get_input_and_target_masks(convs.assistant_masks)
    assert input_mask.shape == target_mask.shape
    input_len = input_mask.sum(dim=-1)
    assert (input_len == input_len[0]).all()
    input_len = th.arange(input_mask.shape[1])[input_mask[0]].max()
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


# todo: kl if you patch from base layers after the probed layer

# %%
adv_samples = entries["Gun adv gun"]
probe_score = patching_exp(
    adv_samples,
    word_to_probe["gun"],
    source_model=nns_base_model,
    target_model=nns_self_obf_model,
)
print(f"Probe score: {probe_score}")

# %%
