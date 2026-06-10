# vvm

Seed repo for a C++ Vector Virtual Machine experiment.

The first target is a compact VVM core:

```text
state -> dot ops -> top candidates -> sample one op -> mul-add -> activation -> normalize -> next state
```

There is no separate program counter, no key/value memory split, and no runtime
writes to the op table in v0. The state vector is both the active computation
and the address used to retrieve the next computation.

## Build

Headless core build:

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

SDL3 visualizer build:

```sh
cmake --preset dev-sdl3
cmake --build --preset dev-sdl3
./build-sdl3/vvm visualize
```

## CLI

```sh
./build/vvm smoke
./build/vvm run --steps 8 --state-dim 256 --ops 1024 --candidates 8 --activation deadzone --update-scale 1.0
./build/vvm run --steps 8 --state-heat 0.01 --op-heat 0.001 --heat-decay 0.999
./build/vvm train-task --task copy-input --activation deadzone --vectors signed --epochs 100 --sample-frames 8 --idle-frames 8 --window 8 --lr 0.1 --rejection-scale 0.01
./build/vvm train-task --task basis-4 --state-dim 16 --ops 16 --candidates 8 --epochs 30 --lr 0.1 --rejection-scale 0.05
./build/vvm train-readout --task linear-2 --state-dim 16 --ops 16 --epochs 10 --readout-source input --readout-lr 0.1
./build/vvm bench-tasks --activation deadzone --vectors signed --epochs 50 --state-dim 16 --ops 64 --candidates 4 --train-samples 16 --test-samples 8 --sample-frames 8 --window 8 --lr 0.1
./scripts/fetch_mnist.sh
./build/vvm train-task --task mnist-01 --state-dim 794 --ops 16 --candidates 8 --train-samples 512 --test-samples 256 --sample-frames 8 --window 8 --epochs 30 --lr 0.1 --rejection-scale 0.05
./build/vvm train-task --task mnist-01 --state-dim 786 --ops 512 --candidates 32 --train-samples 1024 --test-samples 512 --sample-frames 8 --window 8 --epochs 10 --lr 0.01 --max-grad-norm 0.1 --rejection-scale 0.02
./build/vvm train-task --task mnist-01 --state-dim 786 --ops 256 --candidates 16 --train-samples 512 --test-samples 256 --sample-frames 8 --window 8 --epochs 8 --lr 0.002 --lr-decay 0.85 --max-grad-norm 0.1 --rejection-scale 0.02 --rejection-decay 0.5
./build/vvm train-task --task mnist-01 --state-dim 786 --ops 256 --candidates 16 --train-samples 512 --test-samples 256 --sample-frames 8 --window 8 --epochs 12 --lr 0.01 --max-grad-norm 0.1 --rejection-scale 0.04 --rejection-decay 0.85 --class-loss-weight 512
./build/vvm train-task --task mnist-01 --state-dim 786 --ops 256 --candidates 16 --train-samples 512 --test-samples 256 --sample-frames 8 --window 8 --epochs 12 --lr 0.01 --max-grad-norm 0.1 --rejection-scale 0.04 --rejection-overuse-scale 1.0
./build/vvm train-task --task mnist --state-dim 794 --ops 2048 --candidates 8 --train-samples 512 --test-samples 128 --sample-frames 8 --window 8 --epochs 10 --lr 0.05
./build-sdl3/vvm visualize --steps 256
./build-sdl3/vvm visualize-train --epochs 200 --sample-frames 8 --idle-frames 8 --window 8 --lr 0.1 --rejection-scale 0.01
```

`smoke`, `run`, `train-task`, and `bench-tasks` are headless. `visualize` and
`visualize-train` require `VVM_BUILD_VISUALIZER=ON` and SDL3.

The default activation is `deadzone`, a signed hard-threshold:

```text
y = sign(x) * max(abs(x) - threshold, 0)
```

It keeps the cheap threshold behavior of ReLU without erasing negative state
every tick. `deadzone` is the main VVM architecture path. The other activation
modes are kept as ablation/control modes for sanity checks, not as knobs to tune
per task. `--vectors signed|nonnegative` is also available for baseline
comparison.

Current generated tasks are `copy-input`, `delayed-copy`, `linear-2`, `basis-4`,
`alternating-bit`, `xor`, and `sine-next`. MNIST is available as `--task
mnist-01` and `--task mnist` after running `scripts/fetch_mnist.sh`; the data is
stored under ignored `resources/mnist/`.
`copy-input` shows an input vector for
`--sample-frames` ticks and trains state toward that same vector. `delayed-copy`
keeps ticking with no input for `--idle-frames` and then trains state toward the
original vector. Other idle ticks train on the model's own observed next state
as a self-prediction signal. `--rejection-scale` adds a weak local repulsion
from high-loss query/op matches, so failed ops stop monopolizing the same region
of state space.

MNIST v0 maps each 28x28 image into the first 784 state dimensions and trains a
mixed target: reconstruct the image region as a world-model constraint and set
10 digit registers in `state[784..793]` as a supervised readout constraint. The
whole state still addresses the op bank; the registers are loss/readout
dimensions, not a separate control path. The CLI reports exact `test_accuracy`
for class-like tasks.
Task evaluation uses deterministic nearest-op retrieval so reported test metrics
are stable; training may still sample among top candidates unless
`--hard-retrieval` is passed.
`mnist-01` uses the same image region with a balanced two-class split for
digits 0 and 1. `train-task` also reports heat L2, summed learning update L2,
actual op-bank delta L2 for the epoch, total drift from initialization, selected
op count, max op reuse, normalized op-selection entropy, and per-op pressure.
Pressure is tracked as selection count, heat delta L2 sum, and training delta L2
sum for each op; the CLI prints the highest single op plus top-three
`top_select`, `top_train`, and `top_heat` summaries each epoch. It also reports
retrieval sampling diagnostics: `mean_rank`, `mean_prob`, and
`candidate_entropy`. For class tasks it prints label and prediction counts,
`balanced_accuracy`, `class_margin`, `test_nonclass_loss`, and
`test_class_loss`. `--hard-retrieval` uses the nearest op deterministically
instead of sampling among top candidates.
`--class-start-frame`, `--class-value-scale`, `--class-loss-weight`,
`--retrieval-temperature`, `--lr-decay`, `--momentum`, and `--rejection-decay`
are tuning knobs for register targets, candidate sampling, and anti-collapse
pressure. `--class-start-frame` delays class-register loss while still training
reconstruction from frame 0. `--retrieval-temperature 0` keeps the default
linear top-k weighting; positive values use softmax over top-k scores.
`--rejection-overuse-scale` makes rejection focus on ops that are over-selected
relative to the current epoch's usage distribution.

## Layout

- `include/vvm/`: public core interfaces.
- `src/model.cpp`: state/op-bank/retrieval/update implementation.
- `src/tasks.cpp`: first synthetic training tasks.
- `src/main.cpp`: CLI entry point.
- `src/visualizer_sdl3.cpp`: SDL3 inspection path.
- `tests/`: deterministic core smoke tests.
- `docs/vvm-seed-spec.md`: seed architecture notes.
- `docs/model-architecture.md`: current model/training architecture.
- `docs/continuous-vvm.md`: continuous-clock, curiosity, heat, and RL notes.
- `docs/first-tasks.md`: first benchmark/task ladder and output socket notes.
- `resources/`: local research references.
