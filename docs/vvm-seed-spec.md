# VVM Seed Spec

The initial machine is:

```text
candidates = top_n(dot(S(t), OpBank))
op = sample_one(candidates)
S(t + 1) = normalize(ReLU(S(t) + update_scale * op))
```

## V0 Constraints

- One state vector.
- One fixed-size trainable op table.
- State queries nearest ops directly.
- Retrieved ops update state through a fused multiply-add and ReLU.
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
candidate_count = 8
steps = 8
update_scale = 1.0
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
- activation mean/std
- accuracy vs steps
- accuracy vs candidate count

The central question is whether increasing recurrent execution steps improves
performance over an encoder/readout baseline.
