from datasets import load_dataset, concatenate_datasets
import random

# Reasoning format tags
reasoning_start = "\u003creasoning\u003e"
reasoning_end = "\u003c/reasoning\u003e"
solution_start = "\u003canswer\u003e"
solution_end = "\u003c/answer\u003e"

SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """\u003cstart_of_turn\u003euser
{system_prompt}

{question}\u003cend_of_turn\u003e
\u003cstart_of_turn\u003emodel"""

def format_openthoughts_for_sft(example):
    """Format OpenThoughts data for SFT phase."""
    # OpenThoughts already has reasoning traces
    # Format: messages list with role/content
    if 'messages' in example:
        # Extract user query and assistant response
        messages = example['messages']
        user_msg = [m for m in messages if m.get('role') == 'user']
        assistant_msg = [m for m in messages if m.get('role') == 'assistant']
        
        if user_msg and assistant_msg:
            question = user_msg[0]['content']
            response = assistant_msg[0]['content']
            
            return {
                'prompt': TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question),
                'completion': response
            }
    return None

def extract_hash_answer(text):
    """Extract answer from GSM8K format (#### answer)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_gsm8k_for_grpo(example):
    """Format GSM8K data for GRPO phase."""
    return {
        'prompts': TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=example['question']
        ),
        'question': example['question'],
        'answer': extract_hash_answer(example['answer'])
    }

# Load datasets
print("Loading OpenThoughts for SFT...")
sft_dataset = load_dataset(
    "open-thoughts/OpenThoughts-114k",
    split="train[:10000]"  # Use 10k samples for SFT
)
sft_dataset = sft_dataset.map(format_openthoughts_for_sft, remove_columns=sft_dataset.column_names)
sft_dataset = sft_dataset.filter(lambda x: x is not None)

print("Loading GSM8K for GRPO...")
grpo_dataset = load_dataset(
    "gsm8k",
    "main",
    split="train"
)
grpo_dataset = grpo_dataset.map(format_gsm8k_for_grpo, remove_columns=grpo_dataset.column_names)

print(f"SFT dataset size: {len(sft_dataset)}")
print(f"GRPO dataset size: {len(grpo_dataset)}")
