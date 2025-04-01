"""Visualization utilities for self-obfuscation experiments."""

import torch as th
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_layer_sweep_results(layer_sweep_results, max_layer, save_dir="results"):
    """Plot layer sweep results for different interventions.

    Args:
        layer_sweep_results: Dictionary of layer sweep results.
        max_layer: Maximum layer index.
        save_dir: Directory to save plots to.

    Returns:
        Plotly figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        x_title="Layer",
        subplot_titles=["Patch Obfuscated → Base", "Patch Base → Obfuscated"],
    )

    # Define styling parameters
    colors = {"adv_gun": "green", "vanilla_gun": "purple"}
    dash_styles = {1: None, 3: "dash", 5: "longdashdot"}
    marker_styles = {1: "circle", 3: "square", 5: "diamond"}

    # Plot layer sweep results
    for col, from_model in enumerate(["obf", "base"]):
        for num_layers in [1, 3, 5]:
            for example_type in ["adv_gun", "vanilla_gun"]:
                data_key = f"{num_layers}_layer_from_{from_model}_{example_type}"
                if data_key in layer_sweep_results:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(max_layer + 1)),
                            y=layer_sweep_results[data_key],
                            mode="lines+markers",
                            name=f"{num_layers} layers - {'Adversarial' if example_type == 'adv_gun' else 'Vanilla'} Gun Score",
                            line=dict(
                                color=colors[example_type], dash=dash_styles[num_layers]
                            ),
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
        width=1200,
        height=800,
    )

    # Set y-axis range to 0-1 for both subplots
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)

    # Save figures
    fig.write_html(f"{save_dir}/layer_sweep_results.html")
    fig.write_image(
        f"{save_dir}/layer_sweep_results.png", width=1200, height=800, scale=2
    )

    return fig


def plot_stitching_results(
    res_adv_gun,
    res_vanilla_gun,
    res_adv_gun_rev,
    res_vanilla_gun_rev,
    layers_5_6_adv_gun,
    layers_5_6_vanilla_gun,
    max_layer,
    save_dir="results",
):
    """Plot results of forward and reverse layer stitching experiments.

    Args:
        res_adv_gun: List of forward stitching results for adversarial gun examples.
        res_vanilla_gun: List of forward stitching results for vanilla gun examples.
        res_adv_gun_rev: List of reverse stitching results for adversarial gun examples.
        res_vanilla_gun_rev: List of reverse stitching results for vanilla gun examples.
        layers_5_6_adv_gun: Result for layers 5-6 for adversarial gun examples.
        layers_5_6_vanilla_gun: Result for layers 5-6 for vanilla gun examples.
        max_layer: Maximum layer index.
        save_dir: Directory to save plots to.

    Returns:
        Plotly figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)

    fig = go.Figure()

    # Forward stitching traces
    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer + 1)),
            y=res_adv_gun,
            mode="lines+markers",
            name="Forward - Adversarial Gun Score",
            line=dict(color="red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer + 1)),
            y=res_vanilla_gun,
            mode="lines+markers",
            name="Forward - Vanilla Gun Score",
            line=dict(color="blue"),
        )
    )

    # Reverse stitching traces
    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer, -1, -1)),
            y=res_adv_gun_rev,
            mode="lines+markers",
            name="Reverse - Adversarial Gun Score",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer, -1, -1)),
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
        x1=max_layer,
        y1=layers_5_6_adv_gun,
        line=dict(
            color="red",
            width=2,
            dash="dot",
        ),
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=layers_5_6_vanilla_gun,
        x1=max_layer,
        y1=layers_5_6_vanilla_gun,
        line=dict(
            color="blue",
            width=2,
            dash="dot",
        ),
    )

    # Add annotations for the intervention lines
    fig.add_annotation(
        x=max_layer,
        y=layers_5_6_adv_gun,
        text="Layer 5-6 Adversarial",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=0,
        font=dict(color="red", size=12),
    )

    fig.add_annotation(
        x=max_layer,
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
        height=800,
        width=1200,
    )

    # Save figures
    fig.write_html(f"{save_dir}/gun_probe_scores_with_layer_stitching.html")
    fig.write_image(
        f"{save_dir}/gun_probe_scores_with_layer_stitching.png",
        width=1200,
        height=800,
        scale=2,
    )

    return fig


def plot_representation_norms(
    input_reps_norms_adv_gun,
    target_reps_norms_adv_gun,
    input_reps_norms_normalized_adv_gun,
    target_reps_norms_normalized_adv_gun,
    input_base_reps_norms_adv_gun,
    input_obf_reps_norms_adv_gun,
    input_reps_norms_vanilla_gun,
    target_reps_norms_vanilla_gun,
    input_reps_norms_normalized_vanilla_gun,
    target_reps_norms_normalized_vanilla_gun,
    input_base_reps_norms_vanilla_gun,
    input_obf_reps_norms_vanilla_gun,
    save_dir="results",
):
    """Plot representation norm analysis.

    Args:
        Various representation norm metrics for adversarial and vanilla gun examples.
        save_dir: Directory to save plots to.

    Returns:
        Matplotlib figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)

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

    # Save figure
    plt.tight_layout()
    plt.savefig(f"{save_dir}/representation_norms_analysis.png", dpi=300)

    return fig


def plot_kl_divergence(
    kl_div_input_adv_gun,
    kl_div_target_adv_gun,
    stat_dist_input_adv_gun,
    stat_dist_target_adv_gun,
    kl_div_input_vanilla_gun,
    kl_div_target_vanilla_gun,
    stat_dist_input_vanilla_gun,
    stat_dist_target_vanilla_gun,
    save_dir="results",
):
    """Plot KL divergence and statistical distance metrics.

    Args:
        Various KL divergence and statistical distance metrics.
        save_dir: Directory to save plots to.

    Returns:
        Plotly figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)

    fig = make_subplots(rows=1, cols=2)

    # Add bars for KL divergence
    fig.add_trace(
        go.Bar(
            x=["Gun adv gun", "Gun vanilla"],
            y=[kl_div_input_adv_gun, kl_div_input_vanilla_gun],
            name="Input KL Divergence",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["Gun adv gun", "Gun vanilla"],
            y=[kl_div_target_adv_gun, kl_div_target_vanilla_gun],
            name="Target KL Divergence",
        ),
        row=1,
        col=1,
    )

    # Add bars for statistical distance
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
        height=800,
        width=1200,
        legend_title="Metric",
        barmode="group",  # Group bars side by side
    )

    # Save figures
    fig.write_html(
        f"{save_dir}/kl_divergence_between_base_and_self_obfuscated_model.html"
    )
    fig.write_image(
        f"{save_dir}/kl_divergence_between_base_and_self_obfuscated_model.png",
        width=1200,
        height=800,
        scale=2,
    )

    return fig


def plot_top_tokens(
    next_token_probs: th.Tensor,
    tokens: list[str | int],
    tokenizer,
    title: str = None,
    use_token_ids: bool = False,
    file: str | None = None,
    save_html: bool = True,
    height: int = 300,
    width: int = 400,
    annot_size: int = 10,
) -> go.Figure:
    """
    Plot the top tokens for each layer using Plotly.

    Args:
        next_token_probs (th.Tensor): Probability tensor of shape (num_layers, num_tokens, vocab_size)
        tokens (list[str | int]): List of tokens to plot
        tokenizer: Tokenizer object
        title (str): Title of the plot
        use_token_ids (bool): If True, use token IDs instead of token strings
        file (str, optional): File path to save the plot
        save_html (bool): If True, save an HTML file along with the image

    Returns:
        go.Figure: Plotly figure object
    """

    # Ensure next_token_probs has the correct shape
    num_layers, num_tokens, vocab_size = next_token_probs.shape
    assert num_tokens == len(
        tokens
    ), f"Mismatch between number of tokens and number of tokens in list: {num_tokens} != {len(tokens)}"

    def get_top_tokens(probs: th.Tensor) -> tuple:
        top_tokens = th.max(probs, dim=-1)
        top_probs = top_tokens.values
        top_token_ids = [[str(t.item()) for t in layer] for layer in top_tokens.indices]
        top_token_strings = [
            ["'" + tokenizer.convert_ids_to_tokens(t.item()) + "'" for t in layer]
            for layer in top_tokens.indices
        ]
        hover_text = [
            [
                f"ID: {id}<br>Token: {token}"
                for id, token in zip(layer_ids, layer_tokens)
            ]
            for layer_ids, layer_tokens in zip(top_token_ids, top_token_strings)
        ]
        return top_probs, top_token_strings, top_token_ids, hover_text

    fig = go.Figure()

    top_probs, top_token_strings, top_token_ids, hover_text = get_top_tokens(
        next_token_probs
    )
    if isinstance(tokens[0], int):
        tokens = [tokenizer.convert_ids_to_tokens(token) for token in tokens]

    heatmap = go.Heatmap(
        z=top_probs,
        x=list(range(num_tokens)),
        y=list(range(num_layers)),
        text=top_token_ids if use_token_ids else top_token_strings,
        textfont=dict(size=annot_size),
        texttemplate="%{text}",
        colorscale="RdBu_r",
        colorbar=dict(title="Probability", thickness=15, len=0.9),
        hovertext=hover_text,
        hovertemplate="Layer: %{y}<br>%{hovertext}<br>Probability: %{z}<extra></extra>",
    )
    fig.add_trace(heatmap)
    fig.update_traces(zmin=0, zmax=1)
    fig.update_xaxes(
        title_text="Tokens", tickvals=list(range(num_tokens)), ticktext=tokens
    )
    fig.update_yaxes(title_text="Layers")

    fig.update_layout(
        title=title or "Tokens Heatmap",
        height=height,
        width=width,
        showlegend=False,
    )

    if file:
        if isinstance(file, str):
            file = Path(file)
        if file.suffix != ".html":
            fig.write_image(file, scale=3)
        if save_html or file.suffix == ".html":
            fig.write_html(
                file if file.suffix == ".html" else file.with_suffix(".html")
            )
    fig.show()
    return fig
