# Google Tunix Hackathon - Reasoning Model Training

## Project Overview
Kaggle competition to train Gemma model with reasoning traces using Tunix (JAX-native LLM post-training library).

**Goal**: Teach LLM to output `<reasoning>...</reasoning><answer>...</answer>` format.

## Tech Stack
- **Framework**: Tunix + JAX + Flax
- **Model**: Gemma3 1B-IT (or Gemma2 2B)
- **Method**: GRPO (Group Relative Policy Optimization)
- **Fine-tuning**: LoRA (via qwix)
- **Environment**: Kaggle TPU v3-8 (9hr/session, 20hr/week limit)

## Project Structure
```
├── baseline.py          # Original Tunix GRPO demo (reference)
├── src/
│   ├── config.py        # Hyperparameters & settings
│   ├── data_loader.py   # Dataset loading (GSM8K, Open-Platypus)
│   ├── rewards.py       # Reward functions for GRPO
│   └── train.py         # Main training script
└── data/                # Dataset cache
```

## Key Constraints
- Max output tokens: <1K
- English only
- No tool use, no multimodality
- Single Kaggle TPU session for main submission
- Checkpoint must be loadable via Tunix Gemma code (not safetensors)

## Evaluation Domains
Model will be evaluated on diverse domains (not just math):
- Math, Coding, Basic Science (verifiable)
- Creative writing, Ideation, Summarization (non-verifiable)

## Code Conventions
- Use `#%%` cell markers (Jupytext format)
- Shape assertions: `assert x.shape == (...)`
- Insight comments explaining *why*, not *what*
- Visualize training metrics with matplotlib

## Current Strategy
1. **Data**: Mix verifiable domains (GSM8K + Coding + Science)
2. **Rewards**: Format + Correctness + Reasoning Quality
3. **Training**: GRPO with LoRA, cosine LR schedule
4. **Multi-session**: Save/load checkpoints across sessions (optional +15pts)

## Important Hyperparameters
```python
# LoRA
RANK = 32
ALPHA = 32.0

# GRPO
NUM_GENERATIONS = 4      # G in GRPO paper
BETA = 0.08              # KL penalty
EPSILON = 0.2            # PPO clipping

# Training
LEARNING_RATE = 3e-6
MAX_GRAD_NORM = 0.1
WARMUP_STEPS = 10% of total
```

## Reward Function Design
Rewards should be balanced to avoid reward hacking:
- `match_format_exactly`: +3.0 (exact format)
- `check_answer`: +3.0 (correct answer)
- `reasoning_length_reward`: +1.0 max (sufficient detail)
- `reasoning_keywords_reward`: +1.0 max (structured thinking)

## Development Notes
- Always test reward functions independently before training
- Monitor KL divergence during training (should stay bounded)
- Use greedy decoding for evaluation, temperature=0.9 for training
- Save checkpoints every 500 steps for recovery

## References
- [Tunix GitHub](https://github.com/google/tunix/)
- [Tunix Docs](https://tunix.readthedocs.io/)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - GRPO inspiration
- [GRPO Starter Notebook](https://www.kaggle.com/code/windmaple/grpo-demo-gemma3-1b)

---

## Session & Context Management

### Why This Matters
- Claude Code 세션이 끊기거나 새로운 에이전트가 투입될 수 있음
- 컨텍스트 유실 방지를 위해 **문서화 필수**

### Context Handoff Rules
작업 중 반드시 다음을 기록할 것:

1. **현재 진행 상황** → `docs/PROGRESS.md`에 기록
   - 마지막으로 완료한 작업
   - 현재 진행 중인 작업
   - 다음에 해야 할 작업

2. **실험 결과** → `docs/EXPERIMENTS.md`에 기록
   - 하이퍼파라미터 변경 내역
   - 성능 수치 (accuracy, format accuracy 등)
   - 무엇이 효과 있었고 없었는지

3. **의사결정 로그** → `docs/DECISIONS.md`에 기록
   - 왜 특정 데이터셋을 선택했는지
   - 왜 특정 reward 설계를 했는지
   - 시도했다가 폐기한 아이디어들

### Session Recovery Checklist
새 세션 시작 시 확인할 것:
```
1. [ ] docs/PROGRESS.md 읽기 - 현재 상태 파악
2. [ ] docs/EXPERIMENTS.md 읽기 - 이전 실험 결과 확인
3. [ ] git log 확인 - 최근 변경사항
4. [ ] src/config.py 확인 - 현재 하이퍼파라미터
```

### Agent Collaboration Guidelines
여러 에이전트가 협업할 경우:
- 각 에이전트는 작업 시작 전 PROGRESS.md 확인
- 작업 완료 후 반드시 PROGRESS.md 업데이트
- 충돌 방지: 한 번에 한 파일만 수정
- 큰 변경은 브랜치 사용 권장

---

## Current Progress (Update This!)

### Last Updated: 2024-12-27

### Completed
- [x] 프로젝트 구조 분석
- [x] baseline.py 코드 리뷰
- [x] 대회 전략 브레인스토밍
- [x] CLAUDE.md 생성

### In Progress
- [ ] 상세 구현 계획 수립

### Next Steps
1. 다양한 도메인 데이터셋 조사 (Coding, Science)
2. Reward function 개선안 설계
3. 실험 계획 수립
4. 첫 번째 학습 실행

### Blockers / Questions
- (없음)
