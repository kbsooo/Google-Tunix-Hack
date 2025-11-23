#!/usr/bin/env python3
"""
Generate a complete Kaggle notebook for Google Tunix Hackathon.
This script creates a unified .ipynb file with all components integrated.
"""

import json

# Read the Python files
with open('/Users/kbsoo/Codes/kaggle/tunix-hack/data_loader.py', 'r') as f:
    data_loader_code = f.read()

with open('/Users/kbsoo/Codes/kaggle/tunix-hack/reward_functions.py', 'r') as f:
    reward_code = f.read()

# Create the notebook structure
notebook = {
    "cells": [
        # Cell 1: Title
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Google Tunix Hackathon - Multi-Domain Reasoning Training\\n",
                "\\n",
                "**Strategy**: OpenThoughts + GSM8K with GRPO\\n",
                "**Model**: Gemma 3 1B IT + LoRA\\n",
                "\\n",
                "## Key Improvements\\n",
                "1. Multi-reward system (format + accuracy)\\n",
                "2. Optimized hyperparameters for 9-hour TPU session\\n",
                "3. Enhanced evaluation metrics"
            ]
        },
        
        # Cell 2: Installation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\\n",
                "os.environ[\\"HF_HUB_DISABLE_XET\\"] = \\"1\\"\\n",
                "\\n",
                "!pip install -q kagglehub ipywidgets\\n",
                "!pip install -q tensorflow tensorflow_datasets tensorboardX\\n",
                "!pip install -q transformers grain datasets\\n",
                "!pip install \\"google-tunix[prod]==0.1.3\\"\\n",
                "!pip uninstall -q -y flax\\n",
                "!pip install -U flax"
            ]
        },
        
        # Cell 3: Imports
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import functools\\n",
                "import gc\\n",
                "import os\\n",
                "import re\\n",
                "from pprint import pprint\\n",
                "\\n",
                "from flax import nnx\\n",
                "import grain\\n",
                "import humanize\\n",
                "import jax\\n",
                "import jax.numpy as jnp\\n",
                "import kagglehub\\n",
                "import optax\\n",
                "from orbax import checkpoint as ocp\\n",
                "from pathlib import Path\\n",
                "import qwix\\n",
                "from tqdm.auto import tqdm\\n",
                "from tunix.generate import sampler as sampler_lib\\n",
                "from tunix.models.gemma3 import params, model\\n",
                "from tunix.rl import rl_cluster as rl_cluster_lib\\n",
                "from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner\\n",
                "from tunix.rl.rollout import base_rollout\\n",
                "from tunix.sft import metrics_logger\\n",
                "from datasets import load_dataset"
            ]
        },
        
        # Cell 4: Configuration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====== Model & LoRA ======\\n",
                "LORA_RANK = 32\\n",
                "LORA_ALPHA = 32.0\\n",
                "\\n",
                "# ====== Sharding ======\\n",
                "MESH = [(1, 4), (\\"fsdp\\", \\"tp\\")]\\n",
                "\\n",
                "# ====== GRPO ======\\n",
                "MAX_PROMPT_LENGTH = 256\\n",
                "TOTAL_GENERATION_STEPS = 512\\n",
                "TEMPERATURE = 0.9\\n",
                "TOP_P = 1.0\\n",
                "TOP_K = 50\\n",
                "NUM_GENERATIONS = 4\\n",
                "NUM_ITERATIONS = 1\\n",
                "BETA = 0.08\\n",
                "EPSILON = 0.2\\n",
                "\\n",
                "# ====== Training ======\\n",
                "TRAIN_MICRO_BATCH_SIZE = 2\\n",
                "NUM_BATCHES = 3738\\n",
                "NUM_TEST_BATCHES = 100\\n",
                "EVAL_EVERY_N_STEPS = 10\\n",
                "NUM_EPOCHS = 1\\n",
                "MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * NUM_EPOCHS)\\n",
                "\\n",
                "# ====== Optimizer ======\\n",
                "LEARNING_RATE = 3e-6\\n",
                "B1 = 0.9\\n",
                "B2 = 0.99\\n",
                "WEIGHT_DECAY = 0.1\\n",
                "WARMUP_STEPS = int(0.1 * MAX_STEPS)\\n",
                "MAX_GRAD_NORM = 0.1\\n",
                "\\n",
                "# ====== Checkpointing ======\\n",
                "INTERMEDIATE_CKPT_DIR = \\"/tmp/content/intermediate_ckpt/\\"\\n",
                "CKPT_DIR = \\"/tmp/content/ckpts/\\"\\n",
                "SAVE_INTERVAL_STEPS = 500\\n",
                "MAX_TO_KEEP = 4"
            ]
        },
        
        # Cell 5: Data Setup (using data_loader code)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": data_loader_code.split('\\n')
        },
        
        # Cell 6: Reward Functions (using reward_functions code)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": reward_code.split('\\n')
        },
        
        # More cells will be added...
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "tpuV5e8",
            "isInternetEnabled": True,
            "isGpuEnabled": False
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Save the notebook
output_path = '/Users/kbsoo/Codes/kaggle/tunix-hack/tunix_training_complete.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✅ Notebook created: {output_path}")
print(f"Total cells: {len(notebook['cells'])}")
