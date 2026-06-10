# vvm

Seed repo for a C++ Vector Virtual Machine experiment.

The first target is a tiny VVM core:

```text
state -> dot ops -> top candidates -> sample one op -> mul-add -> ReLU -> normalize -> next state
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
./build/vvm run --steps 8 --state-dim 256 --ops 1024 --candidates 8 --update-scale 1.0
./build/vvm run --steps 8 --state-heat 0.01 --op-heat 0.001 --heat-decay 0.999
./build/vvm train-toy --epochs 100 --sample-frames 8 --idle-frames 8 --window 8 --lr 0.1
./build-sdl3/vvm visualize --steps 256
./build-sdl3/vvm visualize-train --epochs 200 --sample-frames 8 --idle-frames 8 --window 8 --lr 0.1
```

`smoke`, `run`, and `train-toy` are headless. `visualize` and
`visualize-train` require `VVM_BUILD_VISUALIZER=ON` and SDL3.

The first toy task is `copy-input`: a nonnegative input vector is shown for
`--sample-frames` ticks and the model trains state toward that same vector.
When `--idle-frames` is nonzero, the model keeps ticking with no input between
samples and trains on its own observed next state as a self-prediction signal.

## Layout

- `include/vvm/`: public core interfaces.
- `src/model.cpp`: state/op-bank/retrieval/update implementation.
- `src/toy_training.cpp`: first synthetic training tasks.
- `src/main.cpp`: CLI entry point.
- `src/visualizer_sdl3.cpp`: SDL3 inspection path.
- `tests/`: deterministic core smoke tests.
- `docs/vvm-seed-spec.md`: seed architecture notes.
- `docs/model-architecture.md`: current model/training architecture.
- `docs/continuous-vvm.md`: continuous-clock, curiosity, heat, and RL notes.
- `resources/`: local research references.
