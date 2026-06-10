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
./build/vvm bench-tasks --activation deadzone --vectors signed --epochs 50 --state-dim 16 --ops 64 --candidates 4 --train-samples 16 --test-samples 8 --sample-frames 8 --window 8 --lr 0.1
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
every tick. `--activation relu|leaky-relu|clamp|deadzone` and
`--vectors signed|nonnegative` are available for comparison.

Current generated tasks are `copy-input`, `delayed-copy`, `alternating-bit`,
`xor`, and `sine-next`. `copy-input` shows an input vector for
`--sample-frames` ticks and trains state toward that same vector. `delayed-copy`
keeps ticking with no input for `--idle-frames` and then trains state toward the
original vector. Other idle ticks train on the model's own observed next state
as a self-prediction signal. `--rejection-scale` adds a weak local repulsion
from high-loss query/op matches, so failed ops stop monopolizing the same region
of state space.

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
