# First VVM Tasks

This is the first task ladder for proving that VVM learns useful computation
before moving to larger datasets.

## Output Model

The VVM core has one recurrent state:

```text
S_t
```

That state should remain the machine state and fetch pointer. External behavior
should be read from state through output sockets/registers:

```text
state -> visual register
state -> audio register
state -> control register
state -> token register
state -> scalar/class register
```

An output socket can start as a fixed or learned projection:

```text
output = readout(S_t)
```

Examples:

```text
visual register: pixels, low-res image, feature grid
audio register: waveform chunk, spectrogram slice, event vector
control register: mouse dx/dy, button logits, key logits
token register: next byte/token logits
class register: category logits
```

This means the state does not literally have to equal the outside world. The
state only has to contain enough information for sockets to read the relevant
parts. Early tasks may train state directly toward a target vector because that
is the simplest test of op-bank learning, but that is not the final interface.

## Observation Model

Inputs should also be typed, but folded into state:

```text
state = normalize(state + input_scale * input_socket_vector)
```

For v0, input vectors can be the same dimension as state. Later, each input
socket can have a tiny encoder/projection into state space:

```text
visual input -> state delta
audio input -> state delta
control feedback -> state delta
token input -> state delta
```

The VM still only runs one state and one op bank.

## Training Target Types

Use three target styles:

```text
state target:
    train predicted state toward a target vector

readout target:
    train a socket/readout sampled from state

next observation target:
    train the model to predict the next external observation or socket value
```

State targets are useful for early debugging. Readout targets are closer to
real control and perception. Next observation targets are the core world-model
training signal.

## Task Ladder

### 1. Copy Input Vector

Show a vector for N frames and train the state or readout to reproduce it.

Purpose:

- proves the op table can adapt
- proves repeated input presentation works
- gives an immediate loss curve

Current repo has a first version of this as `train-task`.

### 2. Delayed Copy

Show a vector for N frames, then hide it for M idle frames. Train a readout to
reproduce the original vector after the delay.

Purpose:

- tests working memory
- tests idle self-dynamics
- catches state collapse after input disappears

### 3. Simple Sequence Prediction

Use low-dimensional sequences:

```text
alternating bit
periodic one-hot symbols
sine wave
sawtooth
small Markov chain
```

Purpose:

- tests temporal prediction
- tests whether more VVM ticks help
- gives easy visual plots

### 4. XOR And Parity

Feed tiny binary vectors and train class output sockets.

Purpose:

- tests nonlinear composition
- separates memorization from simple linear readout

Current task names:

```text
linear-2
basis-4
xor
```

First 16-op no-socket results:

```text
basis-4 reaches 100% accuracy quickly
xor reaches roughly 75% before destabilizing
linear-2 reaches 100% once written as input-preserving class registers
```

The failed early `linear-2` version used a class-only target vector. That exposed
the key/value conflict directly: the input query and target value lived in
different regions. The current version preserves the input dimensions and writes
the answer into class registers.

### 5. Two Moons / Circles

Use 2D classification with a class socket.

Purpose:

- easy to visualize
- harder than linearly separable data
- quick convergence check

### 6. 8x8 Digits

Use the small handwritten digits dataset:

```text
8 x 8 = 64 inputs
10 classes
```

Purpose:

- first image-like classification task
- much smaller than MNIST
- fast enough for sweeps

### 7. MNIST

Use 28x28 grayscale digits:

```text
784 inputs
10 classes
```

Purpose:

- standard baseline
- tests scaling from tiny visual data
- lets us compare against trivial linear/MLP baselines

Current repo support:

```sh
./scripts/fetch_mnist.sh
./build/vvm train-task --task mnist --state-dim 794 --ops 2048 --candidates 8 \
  --train-samples 512 --test-samples 128 --sample-frames 8 --window 8 \
  --epochs 10 --lr 0.05
```

There is also a binary MNIST probe:

```sh
./build/vvm train-task --task mnist-01 --state-dim 794 --ops 16 --candidates 8 \
  --train-samples 512 --test-samples 256 --sample-frames 8 --window 8 \
  --epochs 30 --lr 0.1 --rejection-scale 0.05
```

This first version is VVM-native rather than a standard classifier head: the
image is embedded into `state[0..783]`, and the target both reconstructs that
image region and writes the label into digit registers at `state[784..793]`.
Exact `test_accuracy` is reported from those registers.

Early result: the loader/training path works, but pure label-prototype MNIST
stayed near random accuracy. That version made the image query and digit target
too dissimilar. The current probe preserves image state and adds digit
registers so the core gets both sample reconstruction and class determination
pressure.

Latest no-socket result: `mnist-01` improves with weighted class-register loss
and more ops. A 512-op, 1024-sample run reached about 76% with broad op usage,
but it is not converged. 10-class MNIST remains unsolved. Linear readout sockets
still solve the same raw inputs easily, so the remaining problem is in the VVM
core retrieval/update dynamics rather than the data loader.

Additional sweep result: stronger class-register loss (`--class-loss-weight
512`) and simple epoch-decayed rejection did not break past the low 70s on the
256-op binary task. No rejection collapses usage and predicts almost all class
0. This points toward smarter anti-collapse pressure rather than just stronger
supervised register loss.

### 8. CartPole Observation Prediction

Before control, train next-observation prediction:

```text
obs_t -> obs_t+1
```

Purpose:

- continuous physical stream
- prepares for action-conditioned prediction
- separates world modeling from reward/control

### 9. Tiny Gridworld Stream

Feed local grid observations and train next observation or goal readout.

Purpose:

- discrete controllable world
- later adds external reward
- good first environment for op-affinity policy experiments

## Suggested Order

```text
copy-input
delayed-copy
sequence prediction
xor/parity
two-moons
8x8 digits
MNIST
CartPole prediction
tiny gridworld
```

Do not skip the small generated tasks. They tell us whether failures are architectural or
just dataset scale.
