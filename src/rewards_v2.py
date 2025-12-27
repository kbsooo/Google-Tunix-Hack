#%%
"""
Enhanced Reward Functions for GRPO Training (V2)
Focuses on anti-reward-hacking and reasoning quality.

Key improvements over baseline:
1. penalty_repetition: Detects repetitive n-grams and sentences
2. reward_answer_coherence: Checks if reasoning leads to the stated answer
3. penalty_length_extremes: Penalizes too short or too long reasoning
"""

import re
from typing import List, Optional

#%%
# ====== Constants ======
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

#%%
# ====== Helper Functions ======

def extract_reasoning(completion: str) -> Optional[str]:
    """
    Extract content between <reasoning> and </reasoning> tags.

    Returns:
        str if found, None otherwise
    """
    pattern = rf"{REASONING_START}(.+?){REASONING_END}"
    match = re.search(pattern, completion, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer(completion: str) -> Optional[str]:
    """
    Extract content between <answer> and </answer> tags.

    Returns:
        str if found, None otherwise
    """
    pattern = rf"{ANSWER_START}(.+?){ANSWER_END}"
    match = re.search(pattern, completion, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def get_ngrams(text: str, n: int = 4) -> List[tuple]:
    """
    Generate n-grams from text.

    Args:
        text: Input text
        n: Size of n-gram (default 4)

    Returns:
        List of n-gram tuples
    """
    words = text.lower().split()
    if len(words) < n:
        return []
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def repetition_rate(text: str, n: int = 4) -> float:
    """
    Calculate n-gram repetition rate.

    Insight: High repetition rate indicates reward hacking
    (model generating repeated phrases to pad length)

    Returns:
        Float between 0.0 (no repetition) and 1.0 (all repeated)
    """
    ngrams = get_ngrams(text, n)
    if len(ngrams) == 0:
        return 0.0
    unique = set(ngrams)
    return 1.0 - (len(unique) / len(ngrams))


#%%
# ====== Reward Functions ======

def penalty_repetition(prompts, completions, **kwargs) -> List[float]:
    """
    Penalize repetitive patterns in reasoning.

    Detects:
    1. High n-gram repetition rate (same phrases repeated)
    2. Identical sentences appearing multiple times

    Score range: -3.5 ~ 0.0

    Scoring:
    - n-gram repetition > 50%: -2.0
    - n-gram repetition > 30%: -1.0
    - Same sentence 3+ times: -1.5
    - Normal: 0.0
    """
    scores = []

    for completion in completions:
        score = 0.0
        reasoning = extract_reasoning(completion)

        # No reasoning tag or too short to analyze
        if reasoning is None or len(reasoning.split()) < 20:
            scores.append(0.0)
            continue

        # Check n-gram repetition rate
        rep_rate = repetition_rate(reasoning, n=4)
        if rep_rate > 0.5:
            score -= 2.0
        elif rep_rate > 0.3:
            score -= 1.0

        # Check for identical sentence repetition
        # Split by common sentence delimiters
        sentences = re.split(r'[.!?]', reasoning)
        sentence_counts = {}

        for s in sentences:
            s_clean = s.strip().lower()
            # Only count meaningful sentences (>10 chars)
            if len(s_clean) > 10:
                sentence_counts[s_clean] = sentence_counts.get(s_clean, 0) + 1

        # Check if any sentence appears 3+ times
        if sentence_counts:
            max_repeat = max(sentence_counts.values())
            if max_repeat >= 3:
                score -= 1.5

        scores.append(score)

    return scores


#%%
def reward_answer_coherence(prompts, completions, **kwargs) -> List[float]:
    """
    Check if the reasoning's conclusion matches the stated answer.

    Insight: Good reasoning should derive the answer step-by-step.
    The last numbers mentioned in reasoning should match the answer.

    Score range: -1.0 ~ +1.5

    Scoring:
    - Last number in reasoning == answer: +1.5
    - Answer appears in last 5 numbers: +1.0
    - No match (inconsistent): -1.0
    - Cannot verify: 0.0
    """
    scores = []

    for completion in completions:
        reasoning = extract_reasoning(completion)
        answer = extract_answer(completion)

        # Cannot verify if either is missing
        if reasoning is None or answer is None:
            scores.append(0.0)
            continue

        # Try to parse answer as number
        try:
            # Clean answer: remove commas, handle common formats
            answer_clean = answer.replace(',', '').strip()
            answer_num = float(answer_clean)
        except ValueError:
            # Non-numeric answer, skip verification
            scores.append(0.0)
            continue

        # Extract all numbers from reasoning
        # Pattern: integers or decimals, including negative
        number_pattern = r'-?\d+(?:\.\d+)?'
        numbers_in_reasoning = re.findall(number_pattern, reasoning)

        if not numbers_in_reasoning:
            scores.append(0.0)
            continue

        # Check last 5 numbers for match
        last_numbers = numbers_in_reasoning[-5:]

        try:
            # Check if last number matches
            last_num = float(last_numbers[-1])
            if abs(last_num - answer_num) < 0.001:
                scores.append(1.5)  # Perfect: last number is the answer
                continue

            # Check if answer appears in last 5 numbers
            for num_str in last_numbers:
                num = float(num_str)
                if abs(num - answer_num) < 0.001:
                    scores.append(1.0)  # Good: answer is in conclusion
                    break
            else:
                # Answer not found in last numbers - inconsistent
                scores.append(-1.0)

        except ValueError:
            scores.append(0.0)

    return scores


#%%
def penalty_length_extremes(prompts, completions, **kwargs) -> List[float]:
    """
    Penalize reasoning that is too short or too long.

    Insight:
    - Too short: likely skipping steps, not showing work
    - Too long: likely padding, reward hacking

    Score range: -1.5 ~ 0.0

    Word count thresholds:
    - < 30 words: -1.5 (too short, no real reasoning)
    - 30-50 words: -0.5 (minimal reasoning)
    - 50-600 words: 0.0 (appropriate length)
    - 600-800 words: -0.5 (getting verbose)
    - > 800 words: -1.5 (too long, likely padding)
    """
    scores = []

    for completion in completions:
        reasoning = extract_reasoning(completion)

        # No reasoning tag at all
        if reasoning is None:
            scores.append(-1.5)
            continue

        word_count = len(reasoning.split())

        if word_count < 30:
            score = -1.5  # Too short
        elif word_count < 50:
            score = -0.5  # Minimal
        elif word_count <= 600:
            score = 0.0   # Appropriate
        elif word_count <= 800:
            score = -0.5  # Verbose
        else:
            score = -1.5  # Too long

        scores.append(score)

    return scores


#%%
# ====== Reward Function List ======

REWARDS_V2 = [
    penalty_repetition,
    reward_answer_coherence,
    penalty_length_extremes,
]

#%%
# ====== Tests ======

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Reward Functions V2")
    print("=" * 60)

    # Test cases
    test_cases = [
        # Good case: proper reasoning with coherent answer
        {
            "name": "Good reasoning",
            "completion": f"{REASONING_START}First, we need to calculate 3 + 5. That gives us 8. Then we multiply by 2, which is 8 * 2 = 16. Therefore, the final answer is 16.{REASONING_END}{ANSWER_START}16{ANSWER_END}",
            "expected": "positive or neutral scores"
        },
        # Bad case: repetitive text
        {
            "name": "Repetitive text",
            "completion": f"{REASONING_START}The answer is calculated. The answer is calculated. The answer is calculated. The answer is calculated. The answer is calculated.{REASONING_END}{ANSWER_START}42{ANSWER_END}",
            "expected": "negative repetition penalty"
        },
        # Bad case: too short
        {
            "name": "Too short",
            "completion": f"{REASONING_START}Easy. 42.{REASONING_END}{ANSWER_START}42{ANSWER_END}",
            "expected": "negative length penalty"
        },
        # Bad case: incoherent (reasoning says 16, answer says 42)
        {
            "name": "Incoherent answer",
            "completion": f"{REASONING_START}Let me calculate step by step. First 3 + 5 = 8. Then 8 * 2 = 16. So the result is 16.{REASONING_END}{ANSWER_START}42{ANSWER_END}",
            "expected": "negative coherence penalty"
        },
        # Edge case: no reasoning tag
        {
            "name": "Missing tags",
            "completion": "The answer is 42",
            "expected": "penalties for missing structure"
        },
    ]

    prompts = ["test"] * len(test_cases)
    completions = [tc["completion"] for tc in test_cases]

    print("\n" + "-" * 60)
    print("Individual Function Results:")
    print("-" * 60)

    for fn in REWARDS_V2:
        scores = fn(prompts, completions)
        print(f"\n{fn.__name__}:")
        for i, (tc, score) in enumerate(zip(test_cases, scores)):
            print(f"  [{tc['name']:20s}] {score:+.2f}")

    print("\n" + "-" * 60)
    print("Total Scores per Test Case:")
    print("-" * 60)

    for i, tc in enumerate(test_cases):
        total = sum(fn(prompts, completions)[i] for fn in REWARDS_V2)
        print(f"  [{tc['name']:20s}] Total: {total:+.2f} (expected: {tc['expected']})")

    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)
