# [Google Tunix Hack - Train a model to show its work](https://www.kaggle.com/competitions/google-tunix-hackathon/overview)

Teach a LLM to reason using Tunix, Google’s new JAX-native library for LLM post-training.

## Overview
Most open-source or open-weight language models can give you an answer. But they typically don’t ‘show their work’ - the steps they went through to arrive at that conclusion in a consistent manner. Here, you’ll use Tunix, Google’s new JAX-native library for LLM post-training, to train a model to show its work by laying out a reasoning trace before landing on an answer.

## Description
Reasoning traces help make language models more capable, trustworthy, transparent, and valuable. But building models that reason well requires more than just good data. It takes open tools, strong foundations, and training approaches that the open-source world doesn’t yet have in one place.

In this hackathon, you’ll start with Google’s open-weight Gemma model (Gemma2 2B or Gemma3 1B), fine-tune it with Tunix on TPU, and teach it how to reason through complex questions. You’ll create a model that not only gets the right answer, but also explains how it got there.

Your solution should include a working training pipeline using Tunix and Gemma. Share your configs, reward function composition, and recipes so others can reproduce your results and build on them.

This contribution can help make step-by-step reasoning more accessible for the entire open-source community and lower the barrier to building transparent, capable, and explainable AI.

Like our parents always said: think first, talk later. Let’s teach our models to do the same.

## Timeline
*Note - Judging period subject to change based on the number of submissions received

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Submission Requirements
A valid submission must contain the following:

1. Kaggle Writeup
    a. Media Gallery
    b. Attached Public Notebook
    c. Attached Public Video

Your final Submission must be made prior to the deadline. Any un-submitted or draft Writeups by the competition deadline will not be considered by the Judges.

To create a new Writeup, click on the "New Writeup" button here. After you have saved your Writeup, you should see a "Submit" button in the top right corner.

Note: If you attach a private Kaggle Resource to your public Kaggle Writeup, your private Resource will automatically be made public after the deadline.

1. Kaggle Writeup
The Kaggle Writeup serves as your project report. This should include a title, subtitle, and a detailed analysis of your submission. You must select a Track for your Writeup in order to submit.

Your Writeup should not exceed 1,500 words. Submissions over this limit may be subject to penalty.

The below assets must be attached to the Writeup to be eligible.

a. Media Gallery
This is where you should attach any images and/or videos associated with your submission. A cover image is required to submit your Writeup.

b. Public Notebook
Your code should be submitted as a public notebook in the Project Files field. Your notebook should be publicly accessible and not require a login or paywall. If you use a private Kaggle Notebook, it will automatically be made public after the deadline.

c. Video
Attach your video to the Media Gallery. Videos should be 3 minutes or less, and should be published to Youtube.

## Evaluation
Submissions are evaluated on the following criteria:

### Notebook quality (35 points)

The notebook is clearly written and detailed with:
- Training data
- Hyperparameters
- Prompt
- Training strategy and techniques.

A Gemma2 starter notebook for math reasoning can be found here (Kaggle-hosted version).

Given limited compute on Kaggle TPUs (9 hours per session, 20 hours per week):
- Max output token <1K is fine
- English-only. Multilinguality is not the focus of this hackathon.
- Tool use is not necessary.
- No multimodality

### Model quality for a single Kaggle session (45 points)

- The fine tuned model is the direct output from the notebook above and runs on a single Kaggle TPU session (9 hours).
- The generated model checkpoint files must be loadable and runnable via the Gemma2 or Gemma3 modeling code in Tunix on Kaggle.
- Model output should follow this format:
<reasoning>model_thinking_trace</reasoning>
<answer>model_answer</answer>

Evaluation will cover both the reasoning trace and the final answer, and be done via a combination of LLM-as-a-judge and human judgment. Eval user queries are held out during the hackathon and may come from a range of verifiable and non-verifiable domains, including but not limited to:
- Creative writing
- Creative ideation
- Summarization
- Math
- Coding
- Basic science
- Other

### Video quality (20 points)

The video is under 3 minutes, is informative, and of instructional value for developers who want to learn about training reasoning models using Tunix.
It accurately reflects the author’s work in the hackathon and is of high production quality.

### Model quality across multiple Kaggle sessions (optional, 15 points)

This is optional but participants who want to push the envelope can:
- Finetune the model across multiple Kaggle TPU sessions by saving and loading the intermediate checkpoints.
- Use private data.

Participants must explicitly provide a Kaggle model name/ID at the end of the notebook as the submission for this item. If none is provided, 0 point is given for this item.

The model files must be loadable and runnable via the Gemma2 or Gemma3 modeling code in Tunix on Kaggle, and should not be safetensors files.

## Dataset Description
Welcome to the Google Tunix Hackathon! Here are some resources you can use to help get you started:

Tunix repository on GitHub: https://github.com/google/tunix/
Tunix documentation website: https://tunix.readthedocs.io/en/latest/index.html
GRPO Starter notebook on Kaggle (Gemma2 2B): https://www.kaggle.com/code/windmaple/grpo-demo-gemma2-2b
GRPO Starter notebook on Kaggle (Gemma3 1B): https://www.kaggle.com/code/windmaple/grpo-demo-gemma3-1b
JAX: https://jax.dev
Flax documentation website: https://flax.readthedocs.io/
Paper references:

DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains
Extrapolating RLVR to General Domains without Verifiers
Bridging Offline and Online Reinforcement Learning for LLMs
