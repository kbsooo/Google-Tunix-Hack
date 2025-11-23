# Complete Training Notebook for Google Tunix Hackathon

이 문서는 `tunix_training_complete.ipynb` 노트북을 생성하기 위한 가이드입니다.

## Notebook 셀 구조

### Cell 1: Title & Overview (Markdown)
```markdown
# Google Tunix Hack - Multi-Domain Reasoning Training

**Author**: [Your Name]
**Strategy**: SFT (OpenThoughts) + GRPO (GSM8K)
**Model**: Gemma 3 1B IT with LoRA

## Approach
1. **Phase 1 - SFT**: Train on OpenThoughts-114k to learn reasoning format
2. **Phase 2 - GRPO**: Improve accuracy on GSM8K with custom rewards
3. **Evaluation**: Measure accuracy, partial accuracy, and format compliance
```

### Cell 2: Installation
```python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

!pip install -q kagglehub ipywidgets
!pip install -q tensorflow tensorflow_datasets tensorboardX
!pip install -q transformers grain datasets
!pip install "google-tunix[prod]==0.1.3"
!pip uninstall -q -y flax && !pip install -U flax
```

### Cell 3: Imports
```python
import functools, gc, re
from pprint import pprint
from flax import nnx
import grain, humanize
import jax, jax.numpy as jnp
import kagglehub, optax
from orbax import checkpoint as ocp
import qwix
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma3 import params, model
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from datasets import load_dataset
```

### Cell 4: Config
```python
# Model & LoRA
MODEL_CP_PATH = params.GEMMA3_1B_IT
LORA_RANK = 32
LORA_ALPHA = 32.0

# Training
TRAIN_MICRO_BATCH_SIZE = 2
NUM_BATCHES = 3738
NUM_TEST_BATCHES = 100
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * NUM_EPOCHS)

# GRPO
NUM_GENERATIONS = 4
NUM_ITERATIONS = 1
BETA = 0.08
EPSILON = 0.2
TEMPERATURE = 0.9
TOP_K = 50
TOP_P = 1.0

# Optimizer
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = int(0.1 * MAX_STEPS)
MAX_GRAD_NORM = 0.1

# Paths
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# Special tokens for reasoning format
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f\"\"\"You are given a problem. Think about the problem and provide your reasoning. Place it between {reasoning_start} and {reasoning_end}. Then, provide the final answer (i.e., just one numerical value) between {solution_start} and {solution_end}.\"\"\"

TEMPLATE = \"\"\"<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model\"\"\"
```

### Cell 5: Data Loading
```python
def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_gsm8k_for_grpo(example):
    return {
        'prompts': TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=example['question']
        ),
        'question': example['question'],
        'answer': extract_hash_answer(example['answer'])
    }

print("Loading GSM8K for training...")
dataset = load_dataset("gsm8k", "main", split="train")
dataset = dataset.map(format_gsm8k_for_grpo, remove_columns=dataset.column_names)
train_dataset = dataset[:NUM_BATCHES]

test_dataset = load_dataset("gsm8k", "main", split="test")
test_dataset = test_dataset.map(format_gsm8k_for_grpo, remove_columns=test_dataset.column_names)
test_dataset = test_dataset[:NUM_TEST_BATCHES]

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
```

### Cell 6: Reward Functions
`reward_functions.py`의 전체 코드를 여기에 복사

### Cell 7: Model Loading
```python
# Load base model
!rm -rf {INTERMEDIATE_CKPT_DIR}/*
!rm -rf {CKPT_DIR}/*

mesh = jax.make_mesh(*[(1, 4), ("fsdp", "tp")])
config = model.ModelConfig.gemma3_1b()
gemma = params.create_model_from_checkpoint(MODEL_CP_PATH, config)
tokenizer = params.create_tokenizer()

# Save intermediate checkpoint
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma)
checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
checkpointer.wait_until_finished()
del gemma, state
gc.collect()
```

### Cell 8: Load Reference & Policy Models
```python
def get_gemma_ref_model(ckpt_path):
    # ... (baseline 코드 참고)

def get_lora_model(base_model, mesh):
    # ... (baseline 코드 참고)

ref_model, mesh, model_config = get_gemma_ref_model(
    ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
)
lora_policy = get_lora_model(ref_model, mesh=mesh)
```

### Cell 9: GRPO Training
```python
# Optimizer
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
    optimizer,
)

# GRPO Config
cluster_config = rl_cluster_lib.ClusterConfig(
    # ... (baseline 참고)
)
grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)

# RL Cluster & Learner
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=ref_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    grpo_config=grpo_config,
)

# Train!
with mesh:
    grpo_trainer.train(train_dataset)
```

### Cell 10: Evaluation
```python
# Load latest checkpoint & evaluate
# ... (baseline의 evaluation 코드 참고)
```

## 노트북 생성 방법

위 내용을 바탕으로 Kaggle에서 직접 노트북을 만들거나,
`data_loader.py`와 `reward_functions.py`의 코드를 복사하여 사용하세요.
