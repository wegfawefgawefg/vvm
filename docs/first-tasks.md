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

State targets are useful for early debugging. They are not meant to divide the
VM into isolated regions where one part addresses and another part writes. They
mean "measure this constraint on these dimensions" while the whole recurrent
state and selected op path receive the gradient/update pressure. Readout targets
are closer to real control and perception. Next observation targets are the core
world-model training signal.

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
image region and constrains digit registers at `state[784..793]`. Exact
`test_accuracy` is reported from those registers.

Early result: the loader/training path works, but pure label-prototype MNIST
stayed near random accuracy. That version made the image query and digit target
too dissimilar. The current probe preserves image state and adds digit-register
constraints so the core gets both sample reconstruction/world-model pressure and
class determination pressure from the same state trajectory. These are measured
constraints on one recurrent signal path. The phrase "image addresses,
class-register writes" describes a bad decomposition to avoid, not the design.
The desired behavior is that every active constraint can shape the selected ops
and the resulting state movement.

Latest no-socket result: `mnist-01` improves with weighted class-register loss,
momentum, and delayed class supervision. The best current 256-op run starts
class-register loss on frame 2, keeps image reconstruction active from frame 0,
and reaches about `80.5%` balanced accuracy on the balanced 0-vs-1 test split.
It is not converged. 10-class MNIST remains unsolved. Linear readout sockets
still solve the same raw inputs easily, so the remaining problem is in the VVM
core retrieval/update dynamics rather than the data loader.

Additional sweep result: stronger class-register loss (`--class-loss-weight
512`) and simple epoch-decayed rejection did not break past the low 70s on the
256-op binary task. No rejection collapses usage and predicts almost all class
0. This points toward smarter anti-collapse pressure rather than just stronger
supervised register loss.

Usage-aware rejection (`--rejection-overuse-scale`) keeps 256-op `mnist-01`
usage broad, with roughly 190-200 selected ops and high entropy late in the run.
It also exposes a rising positive `class_margin`, but accuracy still tends to
sit around the high 60s / low 70s at this scale.

Per-class target weighting was also tested to compensate for the 80.5% run's
lower digit-1 recall. Weighting digit 1 more strongly overcorrected toward
predicting digit 1 and lowered balanced accuracy to about `75-76%`, so class
imbalance is a symptom rather than the main fix.

Retrieval-policy sweep: hard top-1 collapses to a tiny active set and stays near
random. Softmax top-k sampling with tested temperatures `0.02`, `0.05`, and
`0.1` also fails. The default linear top-k weighting works best so far because
it samples a small candidate cloud (`mean_rank` around `1.6`, `mean_prob` around
`0.42`) without spreading credit across too many ops.

Larger-data sweep: the same 256-op setup with 2048 train samples peaked around
`77.5%` balanced accuracy and then drifted toward class bias. More samples alone
do not fix the current route stability problem.

Training-cadence sweep: `--train-interval 2`, `4`, and `8` undertrain at the
original learning rate. Raising learning rate in proportion to the interval
recovers the same `80-81%` ceiling, so the current every-frame window replay is
not the obvious convergence bug.

Class-register-width sweep: widening the no-socket class constraint is useful.
`--class-registers 32 --state-dim 816 --class-start-frame 1` reached `87.5%`
balanced accuracy on `mnist-01`, with much better class-1 recall than the
two-register setup. A 1024/512 run peaked around `85.0%` before route drift
pulled it toward class 0. Widths 8, 16, and 64 were worse in the tested schedule.

With 32 registers on the 1024/512 split, `--lr 0.00025 --lr-decay 0.65`
improves stability: it peaked at `85.7%` and ended around `85.4%`. `--restore-best`
restores that best epoch after training, but it should be treated as an
experiment harness feature rather than a final continuous-learning solution.

Sparse op anchoring is now available for stability probes:

```text
--op-anchor-scale F
--anchor-to-best
```

The anchor is not a dense global decay. It only adds a pullback gradient to ops
selected in the current recurrent window, preserving sparse VM-style updates.
Without `--anchor-to-best`, the reference is the initial random op bank. With
`--anchor-to-best`, the reference becomes the best evaluated bank found so far.

On the 1024/512 `mnist-01` 32-register run, initial anchoring at `0.1` hurt
learning and capped balanced accuracy around `83.6%`. Best-bank anchoring at
`0.1` kept the same `85.7%` peak but held that result through later epochs
instead of drifting down to `85.4%`. That suggests the current failure is not
just insufficient bank movement. Useful routes form, then tiny continued
updates can move the active route boundary; anchoring to the best learned bank
is a practical stability diagnostic, not yet a solution to the accuracy ceiling.

Class-register timing can also be ramped:

```text
--class-ramp-frames N
```

This keeps reconstruction/world-model pressure active from the first frame but
scales class-register target weight up over the requested number of frames after
`--class-start-frame`. A hard late gate, such as class supervision only on
frames 6 and 7, undertrained badly at the current learning rate and stayed near
`40%`. A 7-frame ramp from frame 1 was more stable but peaked lower:
`84.4%` with the default class weight and `84.2%` with matched average class
pressure via `--class-loss-weight 224`. Immediate class pressure plus best-bank
anchoring remains the best current schedule, but the ramp confirms that the
objective timing changes route stability.

Positive route retention is available for diagnostics:

```text
--affinity-retain-scale F
--affinity-retain-threshold F
```

This only updates selected ops. If a selected op's prediction error is below the
threshold, the gradient nudges that op toward the working state that selected it,
making the route easier to reselect later. A first `mnist-01` probe with
`--affinity-retain-scale 0.1 --affinity-retain-threshold 0.025` reduced
balanced accuracy to about `84.2%` and made op 36 more dominant. So the sign and
mechanism work, but naive positive affinity makes routes too sticky instead of
breaking the current ceiling. If this idea is reused, it likely needs underuse or
margin gating rather than unconditional "good op gets stickier" pressure.

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
