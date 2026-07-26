# EGPO: Extragradient Preference Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2503.08942-b31b1b.svg)](https://arxiv.org/abs/2503.08942)
[![Conference](https://img.shields.io/badge/Conference-COLM%202025-blue.svg)](https://colmweb.org/index.html)

EGPO (Extragradient Preference Optimization) is a training algorithm for Nash learning from human feedback (NLHF). This repository contains the official implementation of EGPO, which has been accepted to Conference on Language Modeling (COLM) 2025.

## Overview

EGPO introduces an extragradient-based approach to Nash learning from human feedback, providing a more stable and efficient training method compared to existing approaches. The algorithm addresses the challenges of training language models with human feedback by leveraging extragradient optimization techniques.

## Features

- **Multiple NLHF Algorithms**: Implementation of various Nash learning algorithms including:
  - Extragradient preference optimization (EGPO)
  - Online IPO: version 1 (Online Mirror Descent), version 2
  - Nash-MD
  - Nash-MD-PG

- **Large Language Model Support**: Compatible with various transformer models including:
  - Gemma-2-2B
  - Qwen2.5-1.5B
  - And other Hugging Face models

- **Distributed Training**: Built-in support for DeepSpeed distributed training
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with LoRA
- **Comprehensive Evaluation**: Built-in evaluation and testing frameworks

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU(s)
- Conda or Miniconda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/zhourunlong/EGPO.git
cd EGPO
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate egpo
```


## Quick Start

Below we give a brief guide for running language model benchmarks.
If you are interested in running the numerical simulations, please read the files in `simulations/`.

First of all, switch the working directory into `lms/`.

Optionally, you can set `HF_USR_NAME` and `HF_TOKEN` as environment variables to automatically upload model during training.

**You can modify the model and dataset settings in `lms/train_[sft|pref_model|nlhf].py`.**
Below we use the default setting.

### (Optional) SFT for a reference policy model

This step is optional as you can directly use any off-the-shelf text-generation model as the reference policy model.

```bash
deepspeed train_sft.py
```

### (Optional) Build a ground-truth preference model

We use **sequence classification (binary label)** models here.
This step is also optional as you can directly use any off-the-shelf model as the preference model.

```bash
deepspeed train_pref_model.py
```

### NLHF training

We support two ways to host the preference model:

(1) Host a separate preference model on each GPU.
Each preference model only handles the requests from the GPU it resides, so the communication time overhead is minimal.
However, this will take roughly 10~20 GB GPU memory on each GPU.
To use this method, uncomment `judge = PairJudge(pref_model_name)` and comment `judge = LocalServerJudge(server_address)` in `train_nlhf.py`.

(2) **(Recommended)** Host a centralized preference model on an extra GPU.
This preference model handles all the requests from all the GPUs used to train the policy model.
This will only take less than 30 GB GPU memory on that GPU.
However, this requires gathering and distributing requests, so there's a little communication time overhead.
To use this method, run `python judge_server.py` and set the `server_address` in `train_nlhf.py`.

Next, run NLHF algorithms:
```bash
cd lms

# EGPO
deepspeed --master_port=12345 train_nlhf.py --alg eg

# Online IPO v1
deepspeed --master_port=12345 train_nlhf.py --alg oipo1

# Online IPO v2
deepspeed --master_port=12345 train_nlhf.py --alg oipo2

# Nash MD
deepspeed --master_port=12345 train_nlhf.py --alg nmd

# Nash MD PG
deepspeed --master_port=12345 train_nlhf.py --alg nmdpg
```

### Evaluation

Edit `BASE_NAMES` in `test.py` and run

```bash
python test.py
```

Evaluation results are stored in `lms/eval_results/`.


## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{zhou2025extragradient,
  title={Extragradient preference optimization (egpo): Beyond last-iterate convergence for nash learning from human feedback},
  author={Zhou, Runlong and Fazel, Maryam and Du, Simon S},
  journal={arXiv preprint arXiv:2503.08942},
  year={2025}
}
```
