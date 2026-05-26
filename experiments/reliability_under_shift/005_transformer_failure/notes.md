# 005 — Minimal Transformer Collapse on Tiny Dataset

## Core Idea

This experiment implements a minimal Transformer language model from scratch using PyTorch.

The objective was not performance, but understanding:
- token embeddings
- positional embeddings
- self-attention
- autoregressive text generation
- behavior under extremely small datasets

The model was intentionally trained on an absurdly tiny corpus:

"I Love You"

This creates a useful failure scenario where the Transformer rapidly overfits and collapses into repetitive generation patterns.

---

# Research Question

What happens when a Transformer is trained on an extremely low-entropy dataset?

Can self-attention still learn token structure under severe data scarcity?

---

# Architecture

## Components Implemented

- Token Embeddings
- Positional Embeddings
- Single Self-Attention Head
- Causal Attention Masking
- Linear Language Modeling Head

This is effectively:
- a microscopic GPT-style autoregressive language model
- implemented entirely from scratch

---

# Self-Attention Mechanism

The model computes:

Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V

Where:
- Q = Queries
- K = Keys
- V = Values

Causal masking ensures:
- future tokens remain hidden
- generation stays autoregressive

---

# Dataset

Training text:

"I Love You"

Vocabulary extracted directly from the dataset:

[' ', 'I', 'L', 'Y', 'e', 'o', 'u', 'v']

Vocabulary Size:
8

---

# Training Setup

| Parameter | Value |
|---|---|
| Batch Size | 2 |
| Block Size | 4 |
| Embedding Dimension | 64 |
| Learning Rate | 3e-4 |
| Iterations | 5000 |

GPU Used:
Tesla T4

---

# Observed Behavior

## Loss Collapse

Training loss rapidly approached near-zero values:

2.19 → 0.00039

This indicates:
- near-perfect memorization
- complete overfitting
- minimal uncertainty remaining

---

# Generated Output

Example generation:

Love Yove Yove YoeYYv...

The model learned:
- local token relationships
- repetition structure
- short-range sequence continuation

But failed to learn:
- semantic meaning
- grammatical generalization
- long-context coherence

---

# Key Interpretation

This experiment demonstrates an important Transformer behavior:

Self-attention alone does not create intelligence.

Without:
- sufficient data
- diversity
- scale
- regularization

the model collapses into repetitive statistical memorization.

The Transformer successfully learned:
- character transitions
- local sequence structure

but not:
- abstraction
- meaning
- robust language generation

---

# Failure Signal

The generated text exhibits:
- token looping
- repetitive attractor states
- unstable sequence diversity

This is effectively:
mode collapse in autoregressive generation.

---

# Why This Experiment Matters

Modern large language models appear intelligent because of:
- enormous datasets
- massive parameter counts
- multi-head attention
- deep architectures
- large-scale optimization

This experiment isolates the core mechanism and reveals:
- what self-attention alone can and cannot do
- how Transformers behave under severe data starvation

It serves as:
- an educational minimal GPT implementation
- a demonstration of overfitting dynamics
- a toy example of autoregressive collapse

---

# Limitations

- Single attention head
- Single-layer architecture
- Character-level modeling
- Extremely tiny dataset
- No feedforward block
- No normalization layers
- No dropout
- No multi-head attention

---

# Future Work

- Multi-head self-attention
- Feedforward Transformer blocks
- Layer normalization
- Residual connections
- Larger corpora
- Byte Pair Encoding (BPE)
- Attention visualization
- Transformer interpretability analysis
- Loss landscape analysis under tiny datasets

---

# Repository Context

Part of the broader:

ML Failure Proofs

research archive exploring:
- failure dynamics
- representation collapse
- instability under constraints
- robustness limits
- internal model behavior
