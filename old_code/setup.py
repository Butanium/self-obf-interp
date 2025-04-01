from setuptools import find_namespace_packages, setup

setup(
    name="obf_reps",
    version="0.0",
    description="",
    packages=find_namespace_packages(),
    author="",
    author_email="",
    url="https://github.com/LukeBailey181/obfuscated-representations",
    install_requires=[
        # Base
        "transformers",
        "datasets",
        "accelerate",
        "scikit-learn",
        "hydra-core",
        "sae-lens",
        # Visualization
        "matplotlib",
        "pandas",
        # Logging
        "wandb",
        # Utils
        "tqdm",
        "ipykernel",
        "ipywidgets",
        "jaxtyping",
        "sentencepiece",
        "pytest",
        "pre-commit",
        "seaborn",
    ],
    dependency_links=[
        "git+https://github.com/andyzoujm/representation-engineering.git@main#egg=repe",
        "git+https://github.com/EleutherAI/sae@main#egg=sae",
        "git+https://github.com/dsbowen/strong_reject.git@main#egg=strong_reject",
    ],
)
