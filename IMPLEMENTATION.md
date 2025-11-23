# Google Tunix Hackathon - Implementation Guide

## 프로젝트 구조

```
tunix-hack/
├── AGENT.md                    # Agent 지침
├── README.md                   # 대회 정보
├── brainstorming.md            # 전략 브레인스토밍 (artifact)
├── task.md                     # 작업 체크리스트 (artifact)
├── train_pipeline.ipynb        # **메인 학습 노트북** (Kaggle에서 실행)
├── data_loader.py              # 데이터 로딩 코드
├── reward_functions.py         # GRPO 보상 함수
└── start-with-gemma3-1b-it-tutorial.ipynb  # Baseline 참고
```

## 구현된 코드

### 1. `data_loader.py` - 데이터 파이프라인

**SFT Phase:**
- **Dataset**: `open-thoughts/OpenThoughts-114k` (10k samples)
- **Format**: User query + Reasoning trace
- **용도**: 모델에게 reasoning format을 가르침

**GRPO Phase:**
- **Dataset**: `gsm8k` (main split)
- **Format**: Question + Gold answer
- **용도**: 강화학습을 통한 정답률 향상

### 2. `reward_functions.py` - 보상 함수

GRPO 학습에 사용되는 4가지 보상 함수:

1. **`match_format_exactly`**: `<reasoning>...</reasoning><answer>...</answer>` 형식 정확히 일치 시 +3점
2. **`match_format_approximately`**: 각 태그가 정확히 1번씩 등장하면 +0.5점씩
3. **`check_answer`**: 정답 일치 시 +3점, 근사 일치 시 +0.5~1.5점
4. **`check_numbers`**: 숫자 추출 후 정답 일치 시 +1.5점

## Kaggle Notebook 작성 가이드

`train_pipeline.ipynb`를 Kaggle에서 실행할 때 다음 순서로 cell을 작성하세요:

### Cell 1: Setup (이미 완료)
```python
# 라이브러리 설치
```

### Cell 2: Data Loading
`data_loader.py`의 코드를 복사하여 실행:
- OpenThoughts 10k samples
- GSM8K full train set

### Cell 3: Model Loading
Baseline notebook 참고하여:
```python
from tunix.models.gemma3 import model, params
# Gemma 3 1B IT 로딩
# LoRA 적용 (rank=32, alpha=32.0)
```

### Cell 4: SFT Training (Optional)
시간이 허락한다면 SFT phase 추가:
```python
from tunix.sft import SFTTrainer
# SFT on OpenThoughts
```

### Cell 5: GRPO Training
`reward_functions.py`의 보상 함수 사용:
```python
from tunix.rl.grpo import GRPOLearner
# GRPO 학습
reward_fns = [
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers
]
```

### Cell 6: Evaluation
Baseline처럼 greedy/standard/liberal generation으로 테스트:
```python
# Accuracy, Partial Accuracy, Format Accuracy 계산
```

### Cell 7: Checkpoint Saving
```python
# Kaggle Model로 저장
```

## 다음 단계

1. **Kaggle Notebook에 코드 통합**: `data_loader.py`와 `reward_functions.py`의 코드를 `train_pipeline.ipynb`에 복사
2. **Baseline 학습 코드 참고**: `start-with-gemma3-1b-it-tutorial.ipynb`의 모델 로딩/GRPO 학습 부분 재사용
3. **실행**: Kaggle TPU v3-8에서 실행 (9시간 제한 고려)
4. **Writeup & Video 작성**

## 주요 Hyperparameters

```python
MODEL_NAME = "google/gemma-3-1b-it"
LORA_RANK = 32
LORA_ALPHA = 32.0
LEARNING_RATE = 3e-6
BETA = 0.08  # KL penalty
EPSILON = 0.2  # GRPO clipping
NUM_GENERATIONS = 4  # Group size
```

## 주의사항

- Kaggle 환경에서만 제출 가능 (.ipynb)
- Local은 간단한 테스트용으로만 사용
- TPU v3-8 세션 제한 9시간 내에 학습 완료해야 함
