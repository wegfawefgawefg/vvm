# VVM Model Architecture

This document is the working architecture for the raw C++ VVM implementation.
The goal is to keep the runtime small enough to understand and manually train
before adding any autodiff or tensor framework.

## Core Identity

VVM is a continuously running vector virtual machine.

It is not a feed-forward network evaluated once per input. It is a clocked
machine:

```text
state_t -> fetch candidate ops -> execute one op -> state_t+1
```

The state vector is both:

- the active machine state
- the fetch/query pointer for the next op

There is no separate program counter or frame pointer in v0.

## State

```text
S_t in R^d
```

The state contains working memory, continuation, address, and output substrate.
Inputs are folded into state. Outputs are sampled from state.

Current default:

```text
d = 256
```

The state is normalized after major perturbations. This keeps dot-product
retrieval meaningful and prevents runaway magnitude from becoming the whole
addressing scheme.

## Sockets And Connectors

The VVM core should stay small:

```text
state
op bank
retrieval
tick/update
```

External interfaces are sockets. A socket is an attachable pathway between the
world and the VVM state:

```text
world signal <-> connector <-> state
```

The connector may be trainable, but it is not the core VM. It is a small adapter
for a modality or output surface:

```text
eye connector:      pixels -> state injection
digit connector:    state -> digit scores
text connector:     token <-> state
motor connector:    state -> action registers
reward connector:   scalar reward -> reward/value signal
```

This lets us keep the op bank as the shared transition landscape while giving
each interface a compact way to learn how to write into or read from state.
Trained connectors can be shared or cloned:

```text
left camera  -> shared eye connector -> state
right camera -> shared eye connector -> state
```

or copied and allowed to specialize.

This does not mean MNIST should require a socket to learn anything. The
no-socket MNIST task remains a useful probe: can the core op bank itself keep an
image-like observation in state while making class information readable from that
same state? The socket version is the next cleaner experiment because it measures
loss through an attachable readout without forcing the raw state vector itself to
literally equal the label.

The first no-socket MNIST run exposed a likely architectural conflict rather
than a dataset loader bug:

```text
query: image-like state
target: label-like state
op vector: both address key and update value
```

For copy-style tasks this is fine because the query and target live in nearly
the same region. For MNIST classification, a useful update would be selected by
image similarity but move state toward a digit decision. With one homogeneous op
vector, moving the op toward the digit target can move it away from the image
queries that should retrieve it later. That is the key/value conflict we wanted
to postpone in v0.

Current MNIST probe before adding a learned connector:

```text
state[0..783]    image reconstruction region
state[784..793]  digit registers
```

This is only a target layout, not a separate addressing/writing split. The image
region does not own addressing, and the class registers do not own writing. The
whole state still addresses the next op, the chosen op updates the whole state,
and losses are measured wherever a task applies constraints. Reconstructing the
image region is the world-model constraint. Setting class registers is an
additional supervised constraint on the same signal path. Both losses push
through the same state trajectory and selected op, with dimension weights and
timing controlling how much each constraint contributes.

Current diagnostic ladder:

```text
linear-2       linearly separable 2-class vectors
basis-4        four one-hot basis classes
xor            nonlinear 2-bit classification
mnist-01       MNIST digits 0 and 1
mnist          full 10-class MNIST
```

The no-socket 16-op bank can solve `basis-4`, but it does not solve `linear-2`
or `mnist-01`. Linear readout sockets solve those same raw inputs immediately.
That makes `linear-2` the current smallest failure case for the core trainer.

Update: `linear-2` now uses the same input-preserving register target shape as
MNIST. With that target geometry, the pure 16-op base VM reaches 100% quickly.
The old label-prototype version failed because it asked a homogeneous op vector
to be retrieved by an input-like query while moving state toward a dissimilar
class-only prototype.

Weighted observations are now supported for no-socket state targets. This lets
tasks keep reconstruction pressure on input dimensions while applying stronger
loss to class registers. Binary MNIST improves with more ops, weighted class
registers, momentum, and delayed class supervision. The best current 256-op
`mnist-01` run reached about `80.5%` balanced accuracy with class loss starting
on frame 2 while reconstruction stayed active from frame 0. It is better, but it
is not solved yet.

Optional truncated BPTT is available through `--bptt`. The default trainer is a
local one-step update over the window, which is intentionally simple but does not
propagate later loss backward through earlier state transitions. BPTT keeps the
same architecture and sampled op choices, but it backpropagates state gradients
through the recorded window so earlier selected ops can receive pressure from
later constraints.

Tuning notes:

- `--class-loss-weight` increases the gradient pressure on class registers.
- `--class-value-scale` changes the class register target amplitude before
  target normalization.
- `--class-start-frame` delays class-register loss within each shown sample
  while keeping reconstruction/world-model pressure active.
- `--lr-decay` reduces learning rate by epoch.
- `--momentum` enables optimizer momentum over op-bank updates.
- `--rejection-decay` reduces rejection pressure by epoch.

Quick sweeps show that raising `--class-loss-weight` from the default binary
MNIST value to 512 does not solve the task by itself. Rejection is still needed
to avoid op collapse, but persistent rejection can drag learned attractors. The
next likely fix is making rejection conditional on local overuse or decaying it
based on measured op entropy rather than blindly by epoch.

Evaluation now reports `test_nonclass_loss` and `test_class_loss` for class
tasks. On `mnist-01`, nonclass/image reconstruction loss is already tiny
(around `0.0011`) while class-register loss stays much larger (around
`0.36-0.40`). The failure is not that the VM cannot preserve the image-like
observation; it is specifically the class-register steering and route stability.

Current best 256-op binary MNIST probe:

```text
--ops 256 --candidates 16 --sample-frames 8 --window 8
--class-start-frame 2
--lr 0.0005 --lr-decay 0.5 --momentum 0.9
--max-grad-norm 0.1 --update-scale 2.0
--rejection-scale 0.02 --rejection-decay 0.5
```

This reached about `80.5%` balanced accuracy and held that level after the
learning rate decayed. The useful setup change is delaying class-register loss
until frame 2 while keeping reconstruction pressure active from frame 0. Frame 1
can hit about `80%` briefly but is less stable; frames 3-4 undertrain the class
registers. Plain SGD under the same basic setup peaked lower or drifted harder.
Larger banks (`512` ops) spread usage but performed worse with this schedule;
smaller banks show a capacity boundary (`128` ops near `78%`, `64` ops near
collapse/random).

Extra class-1 weighting was tested after the 80.5% run because the model
underpredicted digit 1. It overcorrected toward class 1 and dropped balanced
accuracy to about `75-76%`, so the current bottleneck is not simple class prior
weighting.

The gradient path now has a finite-difference direction test for weighted
targets, including op-row renormalization. That test passed, so the current
MNIST bottleneck is less likely to be a simple weighted-loss sign or denominator
bug.

Usage-aware rejection is now available:

```text
--rejection-overuse-scale F
```

When this is positive, rejection is multiplied by how over-selected the chosen
op is relative to uniform epoch usage. On `mnist-01`, this keeps op usage much
broader than no rejection and avoids the extreme class-0 collapse, but still
does not fully solve the task.

Class tasks also report:

```text
class_margin = score(true class) - max score(other classes)
```

Task evaluation uses deterministic nearest-op retrieval even when training uses
sampled top-k retrieval. This makes reported `test_loss`, `test_accuracy`, and
`class_margin` stable convergence metrics. The runtime and training path can
still sample ops.

Class tasks also report `balanced_accuracy`, the mean per-class recall. This is
the main classification metric when labels are imbalanced or predictions are
biased. The `mnist-01` dataset builder now samples a balanced count of zeros and
ones for both train and test splits.

For `mnist-01`, class margin rises under usage-aware rejection even when accuracy
wobbles, so the class registers are learning a weak signal. The remaining issue
is turning that weak register separation into stable convergence.

Class-conditioned route diagnostics report the top selected ops per label:

```text
class_top=[0:op:count,...;1:op:count,...]
route_purity = sum_op max_label_count(op) / labeled_op_selections
```

For two labels, `route_purity` near 0.5 means both labels are using the same op
routes. Values closer to 1.0 mean selected ops are label-specialized.

Current deterministic-eval MNIST-01 probes show an important drift pattern:
useful class behavior appears early, then continued training can reduce
accuracy even as class margin rises. Lower learning rates and LR/rejection
decay preserve the early route longer, which suggests the remaining problem is
stability/overwriting after useful routes form rather than a complete inability
to learn.

Further probes:

- keep homogeneous ops but add a small readout socket to ask whether class
  information is already present in state
- later, test key/value split only if homogeneous ops cannot handle these probes

## Op Bank

```text
OpBank in R^(N x d)
op_i in R^d
```

The op bank is the trainable instruction landscape. Each op vector is both:

- an addressable object found by dot product
- the executed vector used to update state

Current default:

```text
N = 1024
```

Op rows are normalized after initialization, heat, and training updates.

## Tick Pipeline

One clock tick:

```text
working = normalize(S_t + input_scale * input_t)

scores_i = dot(working, op_i)
candidates = top candidate_count scores
chosen = sample_one(candidates)

z = working + update_scale * op_chosen
a = activation(z)
predicted = normalize(a)

observed = normalize(predicted + state_heat)

prediction_error = mse(predicted, observed)
curiosity_reward = curiosity_scale * prediction_error

S_t+1 = observed
```

When heat is zero, `observed == predicted`.

## Candidate Fetch vs Execution

Candidate retrieval is not execution.

The machine may fetch several likely ops:

```text
candidate_count = 8
```

But it executes exactly one:

```text
chosen = sample_one(candidates)
```

The candidate set exists for local stochasticity and exploration. It is also the
small policy surface for future actor-style learning. Unchosen candidates do not
receive learning credit from that tick.

Each training epoch records per-op selection counts:

```text
selected_ops       number of ops selected at least once
max_op_select      highest selection count for any one op
op_entropy         normalized entropy of the selection distribution
```

This catches collapse into a small loop where a few ops monopolize execution
while most of the op bank is ignored.

## Sampling Policy

The first sampler uses cheap score-derived weights, not softmax:

```text
min_score = min(candidate_scores)
w_i = max(score_i - min_score + epsilon, epsilon)
p_i = w_i / sum(w)
chosen_rank = sample_categorical(p)
```

No exponentials are required.

Greedy prediction still uses the best candidate for deterministic helper calls.
The continuous `run` path samples.

## Heat

Heat is noise injected into the machine to keep it exploring and to prevent dead
vector dynamics.

Current knobs:

```text
state_heat_stddev
op_heat_stddev
heat_decay
```

Heat decays by clock:

```text
effective_heat = initial_heat * heat_decay^clock
```

State heat is applied after prediction. Op-bank heat is applied before fetch.
Both are followed by normalization.

The main risk is permanent noise becoming permanent curiosity. The first
implementation reports this; later training should subtract expected heat noise
or reward prediction progress instead of raw error.

Training diagnostics track heat pressure explicitly:

```text
state_heat_l2  L2 norm of state heat noise added during the epoch
op_heat_l2     L2 norm of op-bank heat noise added during the epoch
learn_l2       L2 norm of actual op-bank changes from learning updates
```

Per-op pressure is tracked too:

```text
op_selection_counts[op]  how often the op was executed
op_heat_l2_by_op[op]     summed heat delta L2 applied to that op
op_train_l2_by_op[op]    summed training update L2 applied to that op
```

The CLI prints compact top-three summaries:

```text
top_select=[op:count,...]
top_train=[op:l2,...]
top_heat=[op:l2,...]
```

If heat is comparable to or larger than learning updates for long runs, the
machine may be rattling more than learning. The SDL training view overlays heat
and learning magnitudes so this can be watched over time.

## Prediction And Curiosity

Prediction loss and curiosity currently use the same observed mismatch.

Model training:

```text
minimize prediction_error
```

Curiosity/control:

```text
reward useful surprise
```

Raw v0:

```text
prediction_error = mse(predicted_next, observed_next)
curiosity_reward = curiosity_scale * prediction_error
```

`observed_next` is not limited to heat. The runtime tick first observes the
actual next state after heat, but an environment or trainer can later replace
that observation with the externally grounded target/observation for the same
tick. The helper is:

```cpp
apply_observation(tick, observed_state, curiosity_scale)
```

This recomputes:

```text
prediction_error
curiosity_reward
total_reward
```

So curiosity does not require heat. Heat is just one possible cause of mismatch.
External observations, sample targets, environment transitions, or internal
state perturbations can all create surprise.

As implemented today, curiosity is a measured intrinsic reward scalar. It is
intentionally dormant for v0 experiments: it is logged and available in the tick
cache, but it does not change heat, input scale, update scale, op affinity, or
sampling behavior.

In VVM terms, behavior is op selection:

```text
state/query -> top-k ops -> sampled chosen op
```

So a future curiosity mechanism should primarily bias op selection pressure:

- pull useful/surprising/learnable op matches closer to their query
- alter chosen-path preference through a value/return rule
- leave heat and input gain to a separate homeostatic controller

Current VVM uses prediction error for local op-content training and optional weak
rejection for bad op/query matches. It does not yet use curiosity to make op
choices more likely.

Raw surprise should also not be maximized blindly. That would reward noise,
chaos, and self-generated instability. The desired signal is a mixture:

```text
curiosity = novelty_weight * surprise
          + progress_weight * learnability
```

Where:

```text
surprise = prediction_error
learnability = max(previous_error_baseline - current_error, 0)
```

This keeps the immediate human-like "what was that?" reaction while reducing
the incentive to chase permanently unpredictable noise.

## External Goals

Arbitrary goals enter as sparse external scalar reward.

No external reward is not the same as observed zero reward. Missing reward must
be masked so continuous unsupervised running does not train the machine on fake
zero-reward data.

Tick reward fields:

```cpp
float external_reward = 0.0F;
bool has_external_reward = false;
```

Total reward:

```text
total_reward_t = curiosity_weight * curiosity_reward_t
if has_external_reward:
    total_reward_t += external_weight * external_reward_t
```

Examples:

- correct sequence output
- target reached
- distance to target decreased
- classification correct at readout time
- survival or stability maintained

The external reward stream is separate from prediction loss. Prediction trains
the machine to model transitions. Reward trains it to prefer some transitions.

External reward normalization/baselines should update only when
`has_external_reward` is true. Do not insert missing rewards into statistics as
zeros.

## Value

Value is a prediction of future reward, not a prediction of next state.

```text
V(S_t) ~= expected discounted future total_reward
```

Value target over a truncated window:

```text
G_t = r_t + gamma * r_t+1 + ... + gamma^n * V(S_t+n)
```

Value loss:

```text
value_loss = mse(V(S_t), G_t)
```

Value exists to move delayed reward backward through time. It should be added
after basic next-state learning works.

## Truncated Window

VVM runs continuously, so there are no natural episodes by default. The last `N`
ticks form the training window:

```text
ring buffer of Tick, length N
```

This is the VVM equivalent of a truncated rollout.

Each tick should store at least:

```cpp
struct Tick {
    std::vector<float> state_before;
    std::vector<float> working_state;
    std::vector<std::size_t> candidate_indices;
    std::vector<float> candidate_probs;
    std::size_t chosen_op;
    float chosen_prob;

    std::vector<float> pre_activation;
    std::vector<float> post_activation;
    std::vector<float> predicted_state;
    std::vector<float> observed_state;

    float prediction_error;
    float curiosity_reward;
    float external_reward;
    bool has_external_reward;
    float total_reward;
    float value;
};
```

States alone are not enough. Training needs the chosen op and transition cache.

## Window Weighting

Older ticks in the ring buffer are useful, but they become stale as op vectors
change. Use two separate discounts:

```text
gamma          future reward discount
recency_decay  update trust discount
```

Reward return:

```text
G_t = r_t + gamma * G_t+1
```

Update trust:

```text
age = 0 for newest tick
age = N - 1 for oldest tick
update_weight = recency_decay^age
```

Normalize by sum of weights so larger windows do not automatically create larger
updates.

Suggested starting values:

```text
gamma = 0.99
recency_decay = 0.97
```

## Manual Trainer V0

Start with manual next-state training only.

Trainable parameter:

```text
op_chosen
```

Frozen/discrete:

```text
candidate selection
chosen op index
```

Forward cache for one tick:

```text
working
chosen_op
pre_activation z
post_activation a
predicted = normalize(a)
target = observed_next
```

Loss:

```text
L = mse(predicted, target)
```

Backward:

```text
dL/dpred = 2 * (predicted - target) / d

normalize backward:
    y = x / ||x||
    dL/dx = (g - y * dot(g, y)) / ||x||

activation backward:
    relu:     dL/dz_i = dL/da_i if z_i > 0 else 0
    deadzone: dL/dz_i = dL/da_i if abs(z_i) > threshold else 0

z = working + update_scale * op_chosen:
    dL/dop_chosen += update_scale * dL/dz
```

Then:

```text
clip op gradient
op_chosen -= learning_rate * weighted_gradient
normalize(op_chosen)
```

Only the chosen op updates for that tick.

### Weak Rejection

VVM does not backprop through discrete candidate selection in v0. Candidate
selection is treated as sampled VM control flow.

To stop a bad op from monopolizing the same query, the trainer can add a weak
local rejection term:

```text
if prediction_error > rejection_threshold:
    gradient += rejection_scale * (prediction_error - rejection_threshold) * working
```

Then normal gradient descent does:

```text
op_chosen -= learning_rate * gradient
```

So the chosen op moves slightly away from the query that it failed to serve.
This is intentionally weak and local. It keeps the op table spread over state
space without introducing a policy network, soft attention, or extra symbolic
machinery.

## Gradient Safety

Use:

- recency weighting
- gradient clipping
- per-op gradient accumulation
- averaging by total recency weight for the window
- row normalization after update

Do not also divide by the number of times an op was chosen after dividing by
the total window weight. If one op is selected for two identical ticks, the
weighted-mean gradient should match the one-tick update, not shrink by another
factor of two. A core regression test now covers that invariant.

## Continuous Stability

VVM is meant to run continuously, so the machine must not slowly decay into heat
noise. Normalization alone is not enough; it preserves vector scale but does not
guarantee useful structure.

Use these safeguards:

- Keep separate heat knobs for state and op bank.
- Start with tiny heat and measure drift before increasing it.
- Decay heat over wall-clock or lower it when prediction error stops improving.
- Normalize state and op rows after heat and training updates.
- Clip gradients before applying op updates.
- Maintain surprise baselines so heat does not become an infinite reward source.
- Prefer prediction progress over raw surprise for intrinsic reward.
- Track op usage entropy and chosen-op churn.
- Track state similarity to recent states and checkpoints.
- Keep periodic stable snapshots of op bank for rollback or comparison.

Useful stability metrics:

```text
prediction_error_mean
prediction_error_slope
curiosity_reward_mean
curiosity_reward_slope
state_norm
op_norm_mean
activation_mean
activation_slope
mean dot(S_t, S_t-1)
mean dot(S_t, S_t-k)
chosen_op_entropy
candidate_entropy
op_bank_drift_from_snapshot
rejection_rate
self_loss_mean
```

Heat should not be treated as free creativity. It is an exploration pressure with
a budget. If prediction error remains high but does not become learnable, the
system is probably chasing noise. In that case reduce heat, subtract a noise
baseline, or mask that source from curiosity.

Homeostasis, separate from curiosity, can tune magnitudes like camera auto ISO:

```text
target_activation = small positive band
if activation_mean too low: increase input/update/heat scale slightly
if activation_mean too high: decrease input/update/heat scale slightly
```

First anti-drift rule:

```text
if self_loss is high and not improving:
    reduce heat or rejection scale
```

First curiosity-progress rule:

```text
learnability = max(previous_error_baseline - current_error, 0)
```

This rewards learning progress instead of permanent unpredictability.

## Core RL Machinery

The goal is to take the useful core from actor-critic, PPO, and DQN without
inheriting their usual network shapes.

The useful actor-critic/PPO core:

- train from finite rollout windows
- store old chosen probabilities
- compute bootstrapped reward-to-go
- compute advantage
- use value for delayed credit assignment
- keep policy updates bounded enough that the running machine is not jolted

The useful DQN core:

- estimate future return for available choices
- use bootstrapped targets
- separate immediate reward from expected future reward
- compare candidate actions/ops through a value-like score

For VVM, the candidate set is the action surface:

```text
state -> candidate ops -> chosen op
```

The actor part is the chosen-op sampler. The critic/value part predicts future
reward from state, and later may also score candidate ops.

Candidate-policy update:

```text
advantage_t = G_t - V(S_t)
if advantage_t > 0: make chosen op more likely
if advantage_t < 0: make chosen op less likely
```

Use the PPO stability lesson directly when updating candidate probabilities:

```text
ratio = new_prob(chosen) / old_prob(chosen)
ratio_clipped = clamp(ratio, 1 - eps, 1 + eps)
```

Use the DQN lesson when scoring candidates:

```text
Q(S_t, candidate_op_i) ~= expected return after choosing candidate_op_i
chosen = sample_or_argmax(Q over candidate set)
```

This does not require a conventional DQN architecture. It only requires a
bootstrapped value target over the candidate choices available to VVM.

## Implementation Order

1. Add tick ring buffer and training cache.
2. Implement manual one-step next-state op update.
3. Add recency weighting and gradient clipping.
4. Run the first generated sequence prediction tasks.
5. Add prediction-progress curiosity baseline.
6. Add external reward stream.
7. Add value readout and truncated returns.
8. Add candidate-policy learning only after the model can predict.

## Activation Policy

The main VVM architecture uses `deadzone`:

```text
y = sign(x) * max(abs(x) - threshold, 0)
```

This is a signed hard threshold. Small values near zero are suppressed, strong
positive and negative values survive, and the derivative is either `0` in the
dead band or `1` outside it. This keeps the implementation simple for manual
training while avoiding the main issue with plain ReLU in a recurrent VM:
negative state cannot persist through ticks.

Other activation modes exist only as ablation/control modes:

```text
relu        old nonnegative control
leaky-relu  signed ReLU-like control
clamp       signed linear-ish control
```

They are useful for checking whether a failure is caused by the deadzone choice,
but they should not be swapped per task. Activation is an architecture decision;
tasks are probes.

Modern feedforward nets get a lot of mileage from ReLU because it is cheap,
stable, and avoids sigmoid/tanh saturation. They can still represent signed
effects through negative weights between layers. VVM is different because its
state persists as memory. If every tick clips state to nonnegative values, the
machine loses signed memory unless we add opponent channels or another explicit
signed representation.
