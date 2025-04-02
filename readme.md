All files in the root folder are mine. Notebook_results contains my plots. Data contains my custom gemma chat templates. The rest is from a version of https://github.com/mgm52/self-obfuscation

To run the code you'll need access to private models from Max, so reach out if you want to run it :)

## File Descriptions

- `probe_experiments_notebook.ipynb`: Python notebook for running experiments to analyze self-obfuscation in language models using probing techniques. Includes layer stitching experiments, representation analysis, and visualization.
- `data_utils.py`: Utilities for loading and processing test data for self-obfuscation experiments, including functions to create variations of examples with different adversarial words.
- `utils.py`: General utility functions for self-obfuscation experiments, including functions for patching model activations, extracting token positions, and layer stitching.
- `visualization.py`: Extensive visualization utilities for plotting experimental results, including layer sweep visualization, representation norm plots, KL divergence visualization, and token probability visualizations.
- `probes.py`: Functions for loading and evaluating probes to detect specific concepts in model representations.
- `models.py`: Model loading and patching utilities, including functions to load checkpoints, stitch model layers, and extract model representations.