# %% [markdown]
# # Self-Obfuscation Probe Experiments
#
# This notebook demonstrates how to use the self-obfuscation probe testing toolkit to analyze
# the mechanisms behind self-obfuscation in language models.

# %%
# Setup and imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import modules from the toolkit
from models import (
    load_local_model,
    stitch_model,
    unstitch_model,
    get_model_reps,
    unembed_probs,
)
from probes import load_word_to_probe_dict, get_probe_score, get_dist_between_reps
from utils import layer_sweep, patching_exp, get_first_word_tokens_pos
from data_utils import load_test_data, create_example_variations
from visualization import (
    plot_layer_sweep_results,
    plot_stitching_results,
    plot_representation_norms,
    plot_kl_divergence,
    plot_top_tokens,
)

# Enable automatic reloading of modules
try:
    from IPython import get_ipython

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except NameError:
    # Not running in IPython/Jupyter environment
    print("Not running in IPython/Jupyter environment")

# %% [markdown]
# ## Configuration
#
# First, we'll set up the configuration for our experiments.

# %%
# Define paths and configuration
WORK_DIR = Path(".")

# Using the exact paths from probe_testing_toolkit.py
test_words = [
    "gun",
    "apple",
    "elephant",
    "SQL",
    "cloud",
    "jester",
    "mathematical",
    "samantha",
]

PROBES_CHECKPOINT_PATH = (
    WORK_DIR
    / "probes_20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_finetuned"
)
MODEL_CHECKPOINT_PATH = (
    WORK_DIR
    / "20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_good"
)
TEST_DATA_PATH = (
    WORK_DIR / "conditional_test_data_gemma_2_9b_instruct_20250226_010745.json"
)
USE_RESPONSE_TOKEN_IDS = True
LAYERS_TO_PROBE = [12]

# Create results directory
RESULTS_DIR = WORK_DIR / "notebook_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Define experiment parameters
LAYERS_TO_PROBE = [12]  # Could also get this from the probe metadata
TEST_WORDS = ["gun", "bomb", "elephant"]

# %% [markdown]
# ## Load Resources
#
# Now we'll load the models, probes, and test data needed for our experiments.

# %%
# Load probes
print("Loading probes...")
word_to_probe = load_word_to_probe_dict(PROBES_CHECKPOINT_PATH)
print(f"Loaded probes for words: {list(word_to_probe.keys())}")

# %%
# Load models
print("Loading self-obfuscated model...")
self_obf_model = load_local_model(checkpoint_path=MODEL_CHECKPOINT_PATH)
print("Loading base model...")
base_model = load_local_model()

# %%
# Load test data
print("Loading test data...")
test_data_df = load_test_data(TEST_DATA_PATH)
print(f"Loaded {len(test_data_df)} test examples")

# Create example variations
entries = create_example_variations(test_data_df, words=TEST_WORDS)
print(f"Created {len(entries)} example variations")

# %% [markdown]
# ## 1. Baseline Comparisons
#
# First, let's compare the probe scores between the base and self-obfuscated models
# for different types of examples.

# %%
# Run baseline comparisons
results = {}

for entry_name, entry in entries.items():
    base_score = (
        get_probe_score(word_to_probe["gun"], entry, model=base_model).mean().item()
    )
    obf_score = (
        get_probe_score(word_to_probe["gun"], entry, model=self_obf_model).mean().item()
    )

    results[entry_name] = {
        "base_model_score": base_score,
        "self_obf_model_score": obf_score,
    }

    print(f"{entry_name}:")
    print(f"  Self Obf model: {obf_score:.4f}")
    print(f"  Base model: {base_score:.4f}")
    print("-" * 100)

# %%
# Visualize baseline comparison results
import plotly.graph_objects as go

fig = go.Figure()

# Add bars for base model
fig.add_trace(
    go.Bar(
        x=list(results.keys()),
        y=[results[k]["base_model_score"] for k in results.keys()],
        name="Base Model",
        marker_color="blue",
    )
)

# Add bars for self-obfuscated model
fig.add_trace(
    go.Bar(
        x=list(results.keys()),
        y=[results[k]["self_obf_model_score"] for k in results.keys()],
        name="Self-Obfuscated Model",
        marker_color="red",
    )
)

# Update layout
fig.update_layout(
    title="Gun Probe Scores: Base vs Self-Obfuscated Model",
    xaxis_title="Example Type",
    yaxis_title="Probe Score",
    barmode="group",
    height=500,
    width=1000,
)

fig.write_html(f"{RESULTS_DIR}/baseline_comparison.html")
fig.write_image(
    f"{RESULTS_DIR}/baseline_comparison.png", width=1200, height=800, scale=2
)
fig.show()

# %% [markdown]
# ## 2. Layer Stitching Experiment
#
# Now let's run a layer stitching experiment to analyze where the self-obfuscation
# behavior is localized in the model.

# %%
# Run forward and reverse layer stitching experiments
max_layer = LAYERS_TO_PROBE[0]
gun_adv_gun_examples = entries["Gun adv gun"]
gun_vanilla_examples = entries["Gun vanilla"]

# Forward stitching (replace layers 0 to i from base model)
res_adv_gun = []
res_vanilla_gun = []
for i in range(max_layer + 1):
    stitch_model(self_obf_model, base_model, list(range(i + 1)))
    adv_score = (
        get_probe_score(
            word_to_probe["gun"], gun_adv_gun_examples, model=self_obf_model
        )
        .mean()
        .item()
    )
    vanilla_score = (
        get_probe_score(
            word_to_probe["gun"], gun_vanilla_examples, model=self_obf_model
        )
        .mean()
        .item()
    )
    res_adv_gun.append(adv_score)
    res_vanilla_gun.append(vanilla_score)
    print(f"Layer 0-{i}:")
    print(f"  Adv gun probe score: {adv_score:.4f}")
    print(f"  Vanilla gun probe score: {vanilla_score:.4f}")
    print("-" * 80)

unstitch_model(self_obf_model, None)

# %%
# Reverse stitching (replace layers i to max_layer from base model)
res_adv_gun_rev = []
res_vanilla_gun_rev = []
for i in range(max_layer, -1, -1):
    stitch_model(self_obf_model, base_model, [i])
    adv_score = (
        get_probe_score(
            word_to_probe["gun"], gun_adv_gun_examples, model=self_obf_model
        )
        .mean()
        .item()
    )
    vanilla_score = (
        get_probe_score(
            word_to_probe["gun"], gun_vanilla_examples, model=self_obf_model
        )
        .mean()
        .item()
    )
    res_adv_gun_rev.append(adv_score)
    res_vanilla_gun_rev.append(vanilla_score)
    print(f"Layer {i}-n:")
    print(f"  Adv gun probe score: {adv_score:.4f}")
    print(f"  Vanilla gun probe score: {vanilla_score:.4f}")
    print("-" * 80)

unstitch_model(self_obf_model, None)

# %%
# Special case: layers 5-6
stitch_model(self_obf_model, base_model, [5, 6])
layers_5_6_adv_gun = (
    get_probe_score(word_to_probe["gun"], gun_adv_gun_examples, model=self_obf_model)
    .mean()
    .item()
)
layers_5_6_vanilla_gun = (
    get_probe_score(word_to_probe["gun"], gun_vanilla_examples, model=self_obf_model)
    .mean()
    .item()
)
unstitch_model(self_obf_model, None)

print(f"Layers 5-6:")
print(f"  Adv gun probe score: {layers_5_6_adv_gun:.4f}")
print(f"  Vanilla gun probe score: {layers_5_6_vanilla_gun:.4f}")

# %%
# Plot layer stitching results
fig = plot_stitching_results(
    res_adv_gun=res_adv_gun,
    res_vanilla_gun=res_vanilla_gun,
    res_adv_gun_rev=res_adv_gun_rev,
    res_vanilla_gun_rev=res_vanilla_gun_rev,
    layers_5_6_adv_gun=layers_5_6_adv_gun,
    layers_5_6_vanilla_gun=layers_5_6_vanilla_gun,
    max_layer=max_layer,
    save_dir=str(RESULTS_DIR),
)
fig.show()

# %% [markdown]
# ## 3. Layer Sweep Experiment
#
# Next, let's run a layer sweep experiment to test the impact of replacing different sets
# of consecutive layers.

# %%
# Run layer sweep experiments with different numbers of consecutive layers
layer_sweep_results = {}

for num_layers in [1, 3, 5]:
    print(f"Running layer sweep with {num_layers} consecutive layers...")

    # From obfuscated model to base model (adversarial examples)
    layer_sweep_results[f"{num_layers}_layer_from_obf_adv_gun"] = layer_sweep(
        base_model,
        self_obf_model,
        word_to_probe["gun"],
        entries["Gun adv gun"],
        num_layers,
        max_layer,
    )

    # From base model to obfuscated model (adversarial examples)
    layer_sweep_results[f"{num_layers}_layer_from_base_adv_gun"] = layer_sweep(
        self_obf_model,
        base_model,
        word_to_probe["gun"],
        entries["Gun adv gun"],
        num_layers,
        max_layer,
    )

    # From obfuscated model to base model (vanilla examples)
    layer_sweep_results[f"{num_layers}_layer_from_obf_vanilla_gun"] = layer_sweep(
        base_model,
        self_obf_model,
        word_to_probe["gun"],
        entries["Gun vanilla"],
        num_layers,
        max_layer,
    )

    # From base model to obfuscated model (vanilla examples)
    layer_sweep_results[f"{num_layers}_layer_from_base_vanilla_gun"] = layer_sweep(
        self_obf_model,
        base_model,
        word_to_probe["gun"],
        entries["Gun vanilla"],
        num_layers,
        max_layer,
    )

# %%
# Plot layer sweep results
fig = plot_layer_sweep_results(
    layer_sweep_results=layer_sweep_results,
    max_layer=max_layer,
    save_dir=str(RESULTS_DIR),
)
fig.show()

# %% [markdown]
# ## 4. Representation Analysis
#
# Now let's analyze the differences in representations between the base and self-obfuscated models.

# %%
# Get representation differences for adversarial gun examples
print("Analyzing representations for adversarial gun examples...")
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
) = get_dist_between_reps(entries["Gun adv gun"], base_model, self_obf_model)

# %%
# Get representation differences for vanilla gun examples
print("Analyzing representations for vanilla gun examples...")
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
) = get_dist_between_reps(entries["Gun vanilla"], base_model, self_obf_model)

# %%
# Plot representation norms
fig_norms = plot_representation_norms(
    input_reps_norms_adv_gun=input_reps_norms_adv_gun,
    target_reps_norms_adv_gun=target_reps_norms_adv_gun,
    input_reps_norms_normalized_adv_gun=input_reps_norms_normalized_adv_gun,
    target_reps_norms_normalized_adv_gun=target_reps_norms_normalized_adv_gun,
    input_base_reps_norms_adv_gun=input_base_reps_norms_adv_gun,
    input_obf_reps_norms_adv_gun=input_obf_reps_norms_adv_gun,
    input_reps_norms_vanilla_gun=input_reps_norms_vanilla_gun,
    target_reps_norms_vanilla_gun=target_reps_norms_vanilla_gun,
    input_reps_norms_normalized_vanilla_gun=input_reps_norms_normalized_vanilla_gun,
    target_reps_norms_normalized_vanilla_gun=target_reps_norms_normalized_vanilla_gun,
    input_base_reps_norms_vanilla_gun=input_base_reps_norms_vanilla_gun,
    input_obf_reps_norms_vanilla_gun=input_obf_reps_norms_vanilla_gun,
    save_dir=str(RESULTS_DIR),
)
plt.show()

# %%
# Plot KL divergence and statistical distance
fig_kl = plot_kl_divergence(
    kl_div_input_adv_gun=kl_div_input_adv_gun,
    kl_div_target_adv_gun=kl_div_target_adv_gun,
    stat_dist_input_adv_gun=stat_dist_input_adv_gun,
    stat_dist_target_adv_gun=stat_dist_target_adv_gun,
    kl_div_input_vanilla_gun=kl_div_input_vanilla_gun,
    kl_div_target_vanilla_gun=kl_div_target_vanilla_gun,
    stat_dist_input_vanilla_gun=stat_dist_input_vanilla_gun,
    stat_dist_target_vanilla_gun=stat_dist_target_vanilla_gun,
    save_dir=str(RESULTS_DIR),
)
fig_kl.show()

# %% [markdown]
# ## 5. Token Probability Analysis
#
# Let's analyze token probabilities to understand how the models make predictions.

# %%
# Get a sample entry
sample_entry = entries["Gun adv gun"].iloc[[0]]

# Get model representations
obf_reps = get_model_reps(
    self_obf_model,
    sample_entry,
    layers=list(range(len(self_obf_model.model.model.layers))),
)
base_reps = get_model_reps(
    base_model,
    sample_entry,
    layers=list(range(len(base_model.model.model.layers))),
)

# Get token probabilities
obf_probs = unembed_probs(obf_reps.input_reps[0], self_obf_model)
base_probs = unembed_probs(base_reps.input_reps[0], base_model)

# Get tokens
tokens = base_model.tokenizer.convert_ids_to_tokens(base_reps.input_ids[0].tolist())
tokens = list(filter(lambda x: x != "<|optim-location|>", tokens))

# %%
# Display sample prompt
print("Sample prompt:")
print(sample_entry.iloc[0]["prompt"])
print("\nSample response:")
print(sample_entry.iloc[0]["response"])

# %%
# Plot token probabilities for self-obfuscated model
fig_obf = plot_top_tokens(
    obf_probs.cpu(),
    tokens,
    self_obf_model.tokenizer,
    height=800,
    width=1250,
    title="Self-Obfuscated Model",
    annot_size=10,
)

# Save figures
fig_obf.write_html(f"{RESULTS_DIR}/top_tokens_obf.html")
fig_obf.write_image(
    f"{RESULTS_DIR}/top_tokens_obf.png", width=1200, height=800, scale=2
)
fig_obf.show()

# %%
# Plot token probabilities for base model
fig_base = plot_top_tokens(
    base_probs.cpu(),
    tokens,
    base_model.tokenizer,
    height=800,
    width=1250,
    title="Base Model",
    annot_size=10,
)

# Save figures
fig_base.write_html(f"{RESULTS_DIR}/top_tokens_base.html")
fig_base.write_image(
    f"{RESULTS_DIR}/top_tokens_base.png", width=1200, height=800, scale=2
)
fig_base.show()

# %% [markdown]
# ## 6. Patching Experiment (Still work in progress, not working for now)
#
# If you have a chat template file, you can run the patching experiment to see how
# patching token activations affects probe scores.
#
# Note: This requires NNSight to be installed and a chat template file.

# %%
CHAT_TEMPLATE_PATH = WORK_DIR / "data/gemma_chat_template.jinja"
# Check if chat template exists
if os.path.exists(CHAT_TEMPLATE_PATH):
    try:
        import nnsight
        from nnsight import LanguageModel
        from transformers import AutoTokenizer

        print("Running patching experiment...")

        # Initialize nnsight models
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        tokenizer.padding_side = "right"

        nns_self_obf_model = LanguageModel(self_obf_model.model, tokenizer=tokenizer)
        nns_base_model = LanguageModel(base_model.model, tokenizer=tokenizer)

        # Run patching experiment
        adv_samples = entries["Gun adv gun"]
        probe_score = patching_exp(
            adv_samples,
            word_to_probe["gun"],
            source_model=nns_base_model,
            target_model=nns_self_obf_model,
            tokenizer=base_model.tokenizer,
            max_layer=LAYERS_TO_PROBE[0],
            chat_template_path=CHAT_TEMPLATE_PATH,
        )

        print(f"Patching experiment probe score: {probe_score}")

    except ImportError:
        print("NNSight not installed. Skipping patching experiment.")
else:
    print("Chat template not found. Skipping patching experiment.")
    print(
        f"To run the patching experiment, please provide a chat template at {CHAT_TEMPLATE_PATH}"
    )

# %% [markdown]
# ## Summary
#
# In this notebook, we've demonstrated the main experiments from the self-obfuscation probe testing toolkit:
#
# 1. Baseline comparisons between base and self-obfuscated models
# 2. Layer stitching experiments to localize self-obfuscation behavior
# 3. Layer sweep experiments to analyze the impact of replacing different sets of layers
# 4. Representation analysis to understand differences between models
# 5. Token probability analysis to visualize how models make predictions
# 6. (Optional) Patching experiment to modify specific model behaviors
#
# The results suggest that self-obfuscation is:
#
# - Localized in specific layers of the model
# - More pronounced for adversarial examples than vanilla examples
# - Characterized by significant differences in token probability distributions
#
# For more detailed analysis, check out the generated plots in the `notebook_results` directory.
