"""Experiment configurations and runners for self-obfuscation experiments."""

import os
import torch as th
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import nnsight as nns
from transformers import AutoTokenizer
from nnsight import LanguageModel

from models import load_local_model, stitch_model, unstitch_model, get_model_reps, unembed_probs
from probes import load_word_to_probe_dict, get_probe_score, get_dist_between_reps
from utils import patching_exp, layer_sweep
from data_utils import load_test_data, create_example_variations, get_test_words
from visualization import (
    plot_layer_sweep_results, 
    plot_stitching_results, 
    plot_representation_norms,
    plot_kl_divergence,
    plot_top_tokens
)


class ExperimentConfig:
    """Configuration for self-obfuscation experiments."""
    
    def __init__(
        self,
        work_dir: str,
        probes_checkpoint_path: Optional[str] = None,
        model_checkpoint_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        layers_to_probe: List[int] = None,
        test_words: List[str] = None,
        results_dir: str = "results",
        chat_template_path: str = None,
    ):
        """
        Initialize experiment configuration.
        
        Args:
            work_dir: Base directory for the experiment.
            probes_checkpoint_path: Path to probe checkpoints.
            model_checkpoint_path: Path to model checkpoint.
            test_data_path: Path to test data.
            layers_to_probe: List of layer indices to probe.
            test_words: List of words to test.
            results_dir: Directory to save results in.
            chat_template_path: Path to chat template file.
        """
        self.work_dir = Path(work_dir)
        
        # Set default paths relative to work_dir if not provided
        if probes_checkpoint_path is None:
            self.probes_checkpoint_path = self.work_dir / "probes_checkpoint"
        else:
            self.probes_checkpoint_path = Path(probes_checkpoint_path)
            
        if model_checkpoint_path is None:
            self.model_checkpoint_path = self.work_dir / "model_checkpoint"
        else:
            self.model_checkpoint_path = Path(model_checkpoint_path)
            
        if test_data_path is None:
            self.test_data_path = self.work_dir / "test_data.json"
        else:
            self.test_data_path = Path(test_data_path)
            
        self.layers_to_probe = layers_to_probe or [12]
        self.test_words = test_words or get_test_words()
        self.results_dir = Path(results_dir)
        self.chat_template_path = chat_template_path
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)


class ExperimentRunner:
    """Runner for self-obfuscation experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.word_to_probe = None
        self.self_obf_model = None
        self.base_model = None
        self.test_data_df = None
        self.entries = None
    
    def load_resources(self):
        """Load resources needed for experiments."""
        # Load probes
        self.word_to_probe = load_word_to_probe_dict(self.config.probes_checkpoint_path)
        
        # Load models
        self.self_obf_model = load_local_model(checkpoint_path=self.config.model_checkpoint_path)
        self.base_model = load_local_model()
        
        # Load test data
        self.test_data_df = load_test_data(self.config.test_data_path)
        
        # Create example variations
        self.entries = create_example_variations(self.test_data_df, words=["bomb", "gun", "elephant"])
    
    def run_baseline_comparisons(self):
        """Run baseline comparisons between base and self-obf models."""
        results = {}
        
        for entry_name, entry in self.entries.items():
            base_score = get_probe_score(self.word_to_probe["gun"], entry, model=self.base_model).mean().item()
            obf_score = get_probe_score(self.word_to_probe["gun"], entry, model=self.self_obf_model).mean().item()
            
            results[entry_name] = {
                "base_model_score": base_score,
                "self_obf_model_score": obf_score,
            }
            
            print(f"{entry_name}:")
            print(f"  Self Obf model: {obf_score}")
            print(f"  Base model: {base_score}")
            print("-" * 100)
        
        return results
    
    def run_layer_stitching_experiment(self):
        """Run forward and reverse layer stitching experiments."""
        max_layer = self.config.layers_to_probe[0]
        gun_adv_gun_examples = self.entries["Gun adv gun"]
        gun_vanilla_examples = self.entries["Gun vanilla"]
        
        # Forward stitching (replace layers 0 to i from base model)
        res_adv_gun = []
        res_vanilla_gun = []
        for i in range(max_layer + 1):
            stitch_model(self.self_obf_model, self.base_model, list(range(i + 1)))
            adv_score = get_probe_score(
                self.word_to_probe["gun"], gun_adv_gun_examples, model=self.self_obf_model
            ).mean().item()
            vanilla_score = get_probe_score(
                self.word_to_probe["gun"], gun_vanilla_examples, model=self.self_obf_model
            ).mean().item()
            res_adv_gun.append(adv_score)
            res_vanilla_gun.append(vanilla_score)
            print(f"Layer 0-{i}:")
            print(f"  Adv gun probe score: {adv_score}")
            print(f"  Vanilla gun probe score: {vanilla_score}")
            print("-" * 100)
        
        unstitch_model(self.self_obf_model, None)
        
        # Reverse stitching (replace layers i to max_layer from base model)
        res_adv_gun_rev = []
        res_vanilla_gun_rev = []
        for i in range(max_layer, -1, -1):
            stitch_model(self.self_obf_model, self.base_model, [i])
            adv_score = get_probe_score(
                self.word_to_probe["gun"], gun_adv_gun_examples, model=self.self_obf_model
            ).mean().item()
            vanilla_score = get_probe_score(
                self.word_to_probe["gun"], gun_vanilla_examples, model=self.self_obf_model
            ).mean().item()
            res_adv_gun_rev.append(adv_score)
            res_vanilla_gun_rev.append(vanilla_score)
            print(f"Layer {i}-n:")
            print(f"  Adv gun probe score: {adv_score}")
            print(f"  Vanilla gun probe score: {vanilla_score}")
            print("-" * 100)
        
        unstitch_model(self.self_obf_model, None)
        
        # Special case: layers 5-6
        stitch_model(self.self_obf_model, self.base_model, [5, 6])
        layers_5_6_adv_gun = get_probe_score(
            self.word_to_probe["gun"], gun_adv_gun_examples, model=self.self_obf_model
        ).mean().item()
        layers_5_6_vanilla_gun = get_probe_score(
            self.word_to_probe["gun"], gun_vanilla_examples, model=self.self_obf_model
        ).mean().item()
        unstitch_model(self.self_obf_model, None)
        
        # Plot results
        fig = plot_stitching_results(
            res_adv_gun=res_adv_gun,
            res_vanilla_gun=res_vanilla_gun,
            res_adv_gun_rev=res_adv_gun_rev,
            res_vanilla_gun_rev=res_vanilla_gun_rev,
            layers_5_6_adv_gun=layers_5_6_adv_gun,
            layers_5_6_vanilla_gun=layers_5_6_vanilla_gun,
            max_layer=max_layer,
            save_dir=str(self.config.results_dir),
        )
        
        return {
            "res_adv_gun": res_adv_gun,
            "res_vanilla_gun": res_vanilla_gun,
            "res_adv_gun_rev": res_adv_gun_rev,
            "res_vanilla_gun_rev": res_vanilla_gun_rev,
            "layers_5_6_adv_gun": layers_5_6_adv_gun,
            "layers_5_6_vanilla_gun": layers_5_6_vanilla_gun,
            "figure": fig,
        }
    
    def run_layer_sweep_experiment(self):
        """Run layer sweep experiments with different numbers of consecutive layers."""
        max_layer = self.config.layers_to_probe[0]
        layer_sweep_results = {}
        
        for num_layers in [1, 3, 5]:
            # Sweep through layers, replacing sets of num_layers consecutive layers
            # from obfuscated model to base model
            layer_sweep_results[f"{num_layers}_layer_from_obf_adv_gun"] = layer_sweep(
                self.base_model, self.self_obf_model, self.word_to_probe["gun"], 
                self.entries["Gun adv gun"], num_layers, max_layer
            )
            
            # Sweep through layers, replacing sets of num_layers consecutive layers
            # from base model to obfuscated model
            layer_sweep_results[f"{num_layers}_layer_from_base_adv_gun"] = layer_sweep(
                self.self_obf_model, self.base_model, self.word_to_probe["gun"], 
                self.entries["Gun adv gun"], num_layers, max_layer
            )
            
            # Same for vanilla examples
            layer_sweep_results[f"{num_layers}_layer_from_obf_vanilla_gun"] = layer_sweep(
                self.base_model, self.self_obf_model, self.word_to_probe["gun"], 
                self.entries["Gun vanilla"], num_layers, max_layer
            )
            
            layer_sweep_results[f"{num_layers}_layer_from_base_vanilla_gun"] = layer_sweep(
                self.self_obf_model, self.base_model, self.word_to_probe["gun"], 
                self.entries["Gun vanilla"], num_layers, max_layer
            )
        
        # Plot results
        fig = plot_layer_sweep_results(
            layer_sweep_results=layer_sweep_results,
            max_layer=max_layer,
            save_dir=str(self.config.results_dir),
        )
        
        return {
            "layer_sweep_results": layer_sweep_results,
            "figure": fig,
        }
    
    def run_representation_analysis(self):
        """Run analysis of representation differences between models."""
        # Get representation differences for adversarial gun examples
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
        ) = get_dist_between_reps(self.entries["Gun adv gun"], self.base_model, self.self_obf_model)
        
        # Get representation differences for vanilla gun examples
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
        ) = get_dist_between_reps(self.entries["Gun vanilla"], self.base_model, self.self_obf_model)
        
        # Plot representation norms
        norms_fig = plot_representation_norms(
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
            save_dir=str(self.config.results_dir),
        )
        
        # Plot KL divergence and statistical distance
        kl_fig = plot_kl_divergence(
            kl_div_input_adv_gun=kl_div_input_adv_gun,
            kl_div_target_adv_gun=kl_div_target_adv_gun,
            stat_dist_input_adv_gun=stat_dist_input_adv_gun,
            stat_dist_target_adv_gun=stat_dist_target_adv_gun,
            kl_div_input_vanilla_gun=kl_div_input_vanilla_gun,
            kl_div_target_vanilla_gun=kl_div_target_vanilla_gun,
            stat_dist_input_vanilla_gun=stat_dist_input_vanilla_gun,
            stat_dist_target_vanilla_gun=stat_dist_target_vanilla_gun,
            save_dir=str(self.config.results_dir),
        )
        
        return {
            "norms_fig": norms_fig,
            "kl_fig": kl_fig,
            "adv_gun_metrics": {
                "input_reps_norms": input_reps_norms_adv_gun,
                "target_reps_norms": target_reps_norms_adv_gun,
                "input_reps_norms_normalized": input_reps_norms_normalized_adv_gun,
                "target_reps_norms_normalized": target_reps_norms_normalized_adv_gun,
                "kl_div_input": kl_div_input_adv_gun,
                "kl_div_target": kl_div_target_adv_gun,
                "stat_dist_input": stat_dist_input_adv_gun,
                "stat_dist_target": stat_dist_target_adv_gun,
            },
            "vanilla_gun_metrics": {
                "input_reps_norms": input_reps_norms_vanilla_gun,
                "target_reps_norms": target_reps_norms_vanilla_gun,
                "input_reps_norms_normalized": input_reps_norms_normalized_vanilla_gun,
                "target_reps_norms_normalized": target_reps_norms_normalized_vanilla_gun,
                "kl_div_input": kl_div_input_vanilla_gun,
                "kl_div_target": kl_div_target_vanilla_gun,
                "stat_dist_input": stat_dist_input_vanilla_gun,
                "stat_dist_target": stat_dist_target_vanilla_gun,
            },
        }
    
    def run_token_probability_analysis(self, sample_idx=0):
        """
        Run analysis of token probabilities for a sample.
        
        Args:
            sample_idx: Index of sample to analyze.
            
        Returns:
            Dictionary of results.
        """
        # Get a sample entry
        sample_entry = self.entries["Gun adv gun"].iloc[[sample_idx]]
        
        # Get model representations
        obf_reps = get_model_reps(
            self.self_obf_model,
            sample_entry,
            layers=list(range(len(self.self_obf_model.model.model.layers))),
        )
        base_reps = get_model_reps(
            self.base_model, 
            sample_entry, 
            layers=list(range(len(self.base_model.model.model.layers))),
        )
        
        # Get token probabilities
        obf_probs = unembed_probs(obf_reps.input_reps[0], self.self_obf_model)
        base_probs = unembed_probs(base_reps.input_reps[0], self.base_model)
        
        # Get tokens
        tokens = self.base_model.tokenizer.convert_ids_to_tokens(base_reps.input_ids[0].tolist())
        tokens = list(filter(lambda x: x != "<|optim-location|>", tokens))
        
        # Plot token probabilities
        fig_obf = plot_top_tokens(
            obf_probs.cpu(),
            tokens,
            self.self_obf_model.tokenizer,
            height=800,
            width=1250,
            title="Self-Obfuscated Model",
            annot_size=10,
        )
        fig_base = plot_top_tokens(
            base_probs.cpu(),
            tokens,
            self.self_obf_model.tokenizer,
            height=800,
            width=1250,
            title="Base Model",
            annot_size=10,
        )
        
        # Save figures
        fig_obf.write_html(f"{self.config.results_dir}/top_tokens_obf.html")
        fig_base.write_html(f"{self.config.results_dir}/top_tokens_base.html")
        fig_obf.write_image(f"{self.config.results_dir}/top_tokens_obf.png", width=1200, height=800, scale=2)
        fig_base.write_image(f"{self.config.results_dir}/top_tokens_base.png", width=1200, height=800, scale=2)
        
        return {
            "fig_obf": fig_obf,
            "fig_base": fig_base,
            "tokens": tokens,
            "sample": sample_entry,
        }
    
    def run_patching_experiment(self):
        """Run patching experiment with nnsight."""
        # Initialize nnsight models
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        tokenizer.padding_side = "right"
        
        nns_self_obf_model = LanguageModel(self.self_obf_model.model, tokenizer=tokenizer)
        nns_base_model = LanguageModel(self.base_model.model, tokenizer=tokenizer)
        
        # Run patching experiment
        adv_samples = self.entries["Gun adv gun"]
        probe_score = patching_exp(
            adv_samples,
            self.word_to_probe["gun"],
            source_model=nns_base_model,
            target_model=nns_self_obf_model,
            tokenizer=self.base_model.tokenizer,
            max_layer=self.config.layers_to_probe[0],
            chat_template_path=self.config.chat_template_path,
        )
        
        print(f"Patching experiment probe score: {probe_score}")
        
        return {
            "probe_score": probe_score,
        }
    
    def run_all_experiments(self):
        """Run all experiments."""
        # Load resources
        self.load_resources()
        
        # Run experiments
        baseline_results = self.run_baseline_comparisons()
        layer_stitching_results = self.run_layer_stitching_experiment()
        layer_sweep_results = self.run_layer_sweep_experiment()
        representation_analysis_results = self.run_representation_analysis()
        token_probability_results = self.run_token_probability_analysis()
        
        # Only run patching experiment if chat template is provided
        patching_results = None
        if self.config.chat_template_path:
            patching_results = self.run_patching_experiment()
        
        return {
            "baseline_results": baseline_results,
            "layer_stitching_results": layer_stitching_results,
            "layer_sweep_results": layer_sweep_results,
            "representation_analysis_results": representation_analysis_results,
            "token_probability_results": token_probability_results,
            "patching_results": patching_results,
        } 