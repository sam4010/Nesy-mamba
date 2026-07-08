# NeSy-Mamba: Slot-Gated State Space Models for Interpretable Logical Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository contains the official codebase for **NeSy-Mamba**, a neuro-symbolic sequence architecture that augments selective state space models (Mamba) with a symbolic slot-gating mechanism for interpretable multi-hop logical reasoning.

---

## 📖 Abstract

While Transformer-based models dominate neural theorem proving, their quadratic attention cost ($O(L^2)$) limits scalability to large knowledge bases, and their reasoning traces remain opaque black boxes. 

**NeSy-Mamba** addresses these issues by:
1. Replacing self-attention with Mamba's **linear-time ($O(L)$) selective scan**.
2. Incorporating $K$ symbolic, sigmoid-gated **truth-flag slots** that track rule activations dynamically across the sequence.
3. Introducing an **exponential moving average (EMA) slot gate** with input-dependent memory coefficients ($\alpha$), allowing slots to function as learnable symbolic filters (similar to GRU update gates).
4. Leveraging a **rule-type taxonomy** (7 semantic categories derived from proof trees) for supervised slot learning.

On the ProofWriter benchmark, NeSy-Mamba achieves **$80.3 \pm 0.4\%$ validation accuracy** with only **273K parameters** (460× smaller than 125M-parameter Transformer baselines). Importantly, type-based symbolic supervision serves as a powerful **training stabiliser**, reducing cross-seed accuracy variance by **16×** (from $\pm6.3$--$6.6\%$ down to $\pm0.4\%$).

---

## 🧠 Architecture Overview

NeSy-Mamba is a self-explaining neural network by construction:

```
Input Tokens ──► Token & Pos Emb ──► Mamba Backbone ──► Last-Token Pool ──┐
                                          │                                │
                                          ├──► EMA Slot Gate (K slots) ───► Concat ──► Answer Head (MLP) ──► Prediction
```

* **Mamba Backbone**: 2 stacked Mamba blocks with pre-norm residual connections and RMSNorm.
* **EMA Slot Gate**: Tracks the activation of 7 distinct logical rules step-by-step.
* **Answer Head**: Jointly fuses semantic representations (from last-token pooling) and symbolic truth values (from slots) for final predictions.
* **Differentiable Logic Losses**: Trained using a combined loss:
  $$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_s \mathcal{L}_{\text{slot}} + \lambda_r \mathcal{L}_{\text{rule}} + \lambda_o \mathcal{L}_{\text{ortho}} + \lambda_e \mathcal{L}_{\text{entropy}}$$

---

## 📁 Repository Structure

```
├── nesy_mamba/              # Core model package
│   ├── config.py            # Hyperparameter configurations
│   ├── mamba_block.py       # Pure-PyTorch Mamba backbone (selective SSM)
│   ├── slot_gate.py         # Symbolic slot gate (monotonic/EMA/GRU modes)
│   ├── logic_loss.py        # Differentiable logic losses
│   ├── nesy_mamba.py        # End-to-end model definition
│   ├── data_utils.py        # Dataset loaders and rule-type classifier
│   ├── proof_parser.py      # Proof tree recursive parser
│   ├── metrics.py           # Evaluation metrics (Accuracy, Logic Fidelity, AUC-MoRF)
│   ├── train.py             # Model training script
│   ├── experiments.py       # Automated ablation/compositional suites
│   └── visualize.py         # Plotting scripts for slot trajectories/heatmaps
├── paper/                   # IEEE paper source files
│   ├── main.tex             # Main paper LaTeX source
│   ├── main_humanized.tex   # Alternate/extended LaTeX source
│   ├── references.bib       # Bibliography database
│   ├── architecture.png     # Architecture diagram
│   └── fig_*.tex            # Plot diagrams in TikZ
├── tests/                   # Model and parser test suites
├── v7_analysis.ipynb        # Jupyter notebook for v7 model analysis
├── v8_analysis.ipynb        # Jupyter notebook for v8 model analysis
└── requirements.txt         # Package dependencies
```

---

## 🚀 Getting Started

### Prerequisites

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests

Ensure your installation works by running the test suite:

```bash
pytest tests/
```

### Training

To train the model on the synthetic logical rules dataset (quick test):

```bash
python -m nesy_mamba.train --variant full --dataset synthetic --epochs 50
```

To train the full supervised model on the ProofWriter dataset (requires downloading datasets into the `nesy_mamba/data` folder):

```bash
python -m nesy_mamba.train \
  --variant full \
  --dataset proofwriter \
  --data_dir ./nesy_mamba/data \
  --slot_gate_mode ema \
  --lambda_slot 1.0 \
  --slot_label_mode type \
  --epochs 20 \
  --batch_size 64
```

---

## 📝 Authors

* **Samarth Bhalerao** - *Department of Electronics and Telecommunication, Vishwakarma Institute of Information Technology, Pune, India* (samarth.22211024@viit.ac.in)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
