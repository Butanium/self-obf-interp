"""Probe loading and evaluation functions for self-obfuscation experiments."""

import os
import pickle
import torch as th
from typing import Dict, List, Optional
from pathlib import Path

from obf_reps.metrics import TrainableMetric

from models import get_model_reps


def load_word_to_probe_dict(probe_checkpoint_dir: str) -> Dict[str, TrainableMetric]:
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

    word_to_probe: Dict[str, TrainableMetric] = {}
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


@th.no_grad()
def get_probe_score(probe, entry, model):
    """Calculate probe score for a given example.

    Args:
        probe: The probe to evaluate with.
        entry: Dict-like object with prompt and response.
        model: The model to get representations from.

    Returns:
        Tensor containing probe scores.
    """
    forward_return = get_model_reps(model, entry)
    return probe.predict_example(
        input_reps=forward_return.input_reps,
        target_reps=forward_return.target_reps,
        target_mask=forward_return.loss_mask,
    )

