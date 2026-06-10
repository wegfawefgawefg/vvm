# PNVM Seed Spec

The initial machine is:

```text
S(t + 1) = F(S(t), Retrieve(S(t), OpBank))
```

## V0 Constraints

- One state vector.
- One fixed-size trainable op table.
- State queries nearest ops directly.
- Retrieved ops update state through a shared transition.
- Classification/readout happens from the final state later.
- No separate program counter.
- No separate key/value memory.
- No dynamic memory bank.
- No writes to the op table during execution.
- No explicit symbolic opcodes.

## First Experiment Shape

Recommended MNIST-ish defaults once training is wired in:

```text
state_dim = 256
num_ops = 1024
top_k = 8
steps = 8
temperature = 1.0
optimizer = AdamW
lr = 1e-3
batch_size = 128
```

## Core Diagnostics

- train loss
- test accuracy
- op usage entropy
- number of ops used per epoch
- mean max retrieval score
- state norm mean/std
- gate mean/std
- accuracy vs steps
- accuracy vs top-k

The central question is whether increasing recurrent execution steps improves
performance over an encoder/readout baseline.
