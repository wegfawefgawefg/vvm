# vvm

Seed repo for a C++ Vector Virtual Machine experiment.

The first target is a tiny VVM core:

```text
state -> nearest ops -> gated residual update -> next state
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
./build/vvm run --steps 8 --state-dim 256 --ops 1024 --top-k 8
./build-sdl3/vvm visualize --steps 256
```

`smoke` and `run` are headless. `visualize` requires `VVM_BUILD_VISUALIZER=ON`
and SDL3.

## Layout

- `include/vvm/`: public core interfaces.
- `src/model.cpp`: state/op-bank/retrieval/update implementation.
- `src/main.cpp`: CLI entry point.
- `src/visualizer_sdl3.cpp`: SDL3 inspection path.
- `tests/`: deterministic core smoke tests.
- `docs/vvm-seed-spec.md`: seed architecture notes.
