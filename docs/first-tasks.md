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

Current repo has a first version of this as `train-toy`.

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

Do not skip the tiny tasks. They tell us whether failures are architectural or
just dataset scale.
