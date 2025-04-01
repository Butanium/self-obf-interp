import torch as th
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from pathlib import Path


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


def test_get_input_and_target_masks():
    # Test case 1: Simple alternating pattern
    assistant_mask = th.tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    input_mask, target_mask = get_input_and_target_masks(assistant_mask.unsqueeze(0))

    expected_input = th.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0]).unsqueeze(0)
    expected_target = th.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 0]).unsqueeze(0)

    assert th.all(
        input_mask == expected_input
    ), f"Input mask mismatch: {input_mask} != {expected_input}"
    assert th.all(
        target_mask == expected_target
    ), f"Target mask mismatch: {target_mask} != {expected_target}"

    # Test case 2: Only one sequence
    assistant_mask = th.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    input_mask, target_mask = get_input_and_target_masks(assistant_mask.unsqueeze(0))

    expected_input = th.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0]).unsqueeze(0)
    expected_target = th.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0)

    assert th.all(
        input_mask == expected_input
    ), f"Input mask mismatch: {input_mask} != {expected_input}"
    assert th.all(
        target_mask == expected_target
    ), f"Target mask mismatch: {target_mask} != {expected_target}"

    # Test case 3: Empty tensor
    assistant_mask = th.tensor([0, 0, 0, 0, 0]).unsqueeze(0)
    input_mask, target_mask = get_input_and_target_masks(assistant_mask)

    expected_input = th.tensor([0, 0, 0, 0, 0]).unsqueeze(0)
    expected_target = th.tensor([0, 0, 0, 0, 0]).unsqueeze(0)

    assert th.all(
        input_mask == expected_input
    ), f"Input mask mismatch: {input_mask} != {expected_input}"
    assert th.all(
        target_mask == expected_target
    ), f"Target mask mismatch: {target_mask} != {expected_target}"

    # Test case 4: Multiple sequences
    assistant_mask = th.tensor([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
    input_mask, target_mask = get_input_and_target_masks(assistant_mask.unsqueeze(0))

    expected_input = th.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0)
    expected_target = th.tensor([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]).unsqueeze(0)

    assert th.all(
        input_mask == expected_input
    ), f"Input mask mismatch: {input_mask} != {expected_input}"
    assert th.all(
        target_mask == expected_target
    ), f"Target mask mismatch: {target_mask} != {expected_target}"

    # Test case 5: Batch of masks
    assistant_mask = th.tensor([
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0]
    ])
    input_mask, target_mask = get_input_and_target_masks(assistant_mask)

    expected_input = th.tensor([
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0]
    ])
    expected_target = th.tensor([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0]
    ])

    assert th.all(
        input_mask == expected_input
    ), f"Input mask mismatch: {input_mask} != {expected_input}"
    assert th.all(
        target_mask == expected_target
    ), f"Target mask mismatch: {target_mask} != {expected_target}"

    print("All tests passed!")


if __name__ == "__main__":
    test_get_input_and_target_masks()
