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
a = ReLU(z)
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

## Prediction And Curiosity

Prediction loss and curiosity use the same error signal in opposite ways.

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

Better future signal:

```text
curiosity = previous_expected_error - current_error
```

or:

```text
curiosity = prediction_error - noise_baseline
```

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

    std::vector<float> pre_relu;
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
pre_relu z
post_relu a
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

ReLU backward:
    dL/dz_i = dL/da_i if z_i > 0 else 0

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

## Gradient Safety

Use:

- recency weighting
- gradient clipping
- per-op gradient accumulation
- optional division by number of times an op was chosen in the window
- row normalization after update

Repeated chosen-op hits in one window can otherwise over-update a single vector.

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
state_norm
op_norm_mean
mean dot(S_t, S_t-1)
mean dot(S_t, S_t-k)
chosen_op_entropy
candidate_entropy
op_bank_drift_from_snapshot
```

Heat should not be treated as free creativity. It is an exploration pressure with
a budget. If prediction error remains high but does not become learnable, the
system is probably chasing noise. In that case reduce heat, subtract a noise
baseline, or mask that source from curiosity.

First anti-drift rule:

```text
if prediction_error is high and prediction_error_slope >= 0:
    reduce effective heat
```

First curiosity rule:

```text
curiosity = max(previous_error_baseline - current_error, 0)
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
4. Run tiny sequence prediction tasks.
5. Add prediction-progress curiosity baseline.
6. Add external reward stream.
7. Add value readout and truncated returns.
8. Add candidate-policy learning only after the model can predict.
