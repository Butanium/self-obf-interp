#!/usr/bin/env python3
"""Main entry point for running self-obfuscation experiments."""

import argparse
import os
import json
from pathlib import Path

from experiment import ExperimentConfig, ExperimentRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run self-obfuscation experiments")
    
    parser.add_argument(
        "--work_dir",
        type=str,
        default=Path(__file__).parent,
        help="Base directory for the experiment",
    )
    
    parser.add_argument(
        "--probes_checkpoint_path",
        type=str,
        help="Path to probe checkpoints",
    )
    
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        help="Path to model checkpoint",
    )
    
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test data",
    )
    
    parser.add_argument(
        "--layers_to_probe",
        type=int,
        nargs="+",
        default=[12],
        help="Layer indices to probe",
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results in",
    )
    
    parser.add_argument(
        "--chat_template_path",
        type=str,
        help="Path to chat template file",
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "all",
            "baseline",
            "layer_stitching",
            "layer_sweep",
            "representation_analysis",
            "token_probability",
            "patching",
        ],
        default="all",
        help="Experiment to run",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig(
        work_dir=args.work_dir,
        probes_checkpoint_path=args.probes_checkpoint_path,
        model_checkpoint_path=args.model_checkpoint_path,
        test_data_path=args.test_data_path,
        layers_to_probe=args.layers_to_probe,
        results_dir=args.results_dir,
        chat_template_path=args.chat_template_path,
    )
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Load resources
    print("Loading resources...")
    runner.load_resources()
    
    # Run specified experiment
    if args.experiment == "all":
        print("Running all experiments...")
        results = runner.run_all_experiments()
    elif args.experiment == "baseline":
        print("Running baseline comparisons experiment...")
        results = runner.run_baseline_comparisons()
    elif args.experiment == "layer_stitching":
        print("Running layer stitching experiment...")
        results = runner.run_layer_stitching_experiment()
    elif args.experiment == "layer_sweep":
        print("Running layer sweep experiment...")
        results = runner.run_layer_sweep_experiment()
    elif args.experiment == "representation_analysis":
        print("Running representation analysis experiment...")
        results = runner.run_representation_analysis()
    elif args.experiment == "token_probability":
        print("Running token probability analysis experiment...")
        results = runner.run_token_probability_analysis()
    elif args.experiment == "patching":
        print("Running patching experiment...")
        if config.chat_template_path is None:
            print("Error: chat_template_path is required for patching experiment")
            return
        results = runner.run_patching_experiment()
    
    print("Experiments completed successfully!")


if __name__ == "__main__":
    main() 