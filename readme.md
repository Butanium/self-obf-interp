# Self-Obfuscation Probe Testing Toolkit

This toolkit provides a modular framework for testing and analyzing self-obfuscation behavior in language models using probes.

## Overview

The toolkit allows you to:

1. Compare base and self-obfuscated models using probes
2. Perform layer stitching experiments to analyze localization of self-obfuscation behavior
3. Analyze representation differences between models
4. Visualize token probabilities and other metrics
5. Run patching experiments to modify specific model behaviors

## Project Structure

- `main.py`: Entry point for running experiments
- `models.py`: Model loading and patching utilities
- `probes.py`: Probe loading and evaluation functions
- `visualization.py`: Plotting and visualization utilities
- `utils.py`: General utility functions
- `experiment.py`: Experiment configurations and runners
- `data_utils.py`: Data loading and processing functions

## Usage

### Basic Usage

```bash
python main.py --probes_checkpoint_path <path-to-probes> \
               --model_checkpoint_path <path-to-model> \
               --test_data_path <path-to-test-data> \
               --results_dir results
```

### Running Specific Experiments

```bash
# Run only the layer stitching experiment
python main.py --experiment layer_stitching \
               --probes_checkpoint_path <path-to-probes> \
               --model_checkpoint_path <path-to-model> \
               --test_data_path <path-to-test-data>

# Run only the representation analysis experiment
python main.py --experiment representation_analysis \
               --probes_checkpoint_path <path-to-probes> \
               --model_checkpoint_path <path-to-model> \
               --test_data_path <path-to-test-data>
```

### Patching Experiment

For patching experiments, you need to provide a chat template file:

```bash
python main.py --experiment patching \
               --probes_checkpoint_path <path-to-probes> \
               --model_checkpoint_path <path-to-model> \
               --test_data_path <path-to-test-data> \
               --chat_template_path <path-to-chat-template>
```

## Experiments

### Baseline Comparisons

Compare probe scores between base and self-obfuscated models for different types of examples.

### Layer Stitching

Replace layers between models in different patterns to analyze where self-obfuscation behavior is localized.

### Layer Sweep

Systematically replace consecutive layers to analyze the impact on probe scores.

### Representation Analysis

Analyze differences in representations between base and self-obfuscated models using various metrics.

### Token Probability Analysis

Visualize token probabilities to understand how models are making predictions.

### Patching Experiment

Use NNSight to patch activations between models and analyze the effect on probe scores.

## Results

Experiment results are saved to the specified results directory (default: `results/`). This includes:

- HTML and PNG visualizations
- Layer stitching and sweep plots
- Representation norm analysis
- KL divergence and statistical distance metrics
- Token probability heatmaps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
