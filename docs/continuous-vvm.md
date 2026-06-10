# Continuous VVM Notes

VVM should be treated as a continuously running machine, not a one-shot
feed-forward model.

```text
clock tick:
    input is folded into state
    state queries op bank by dot product
    top candidates are fetched
    one candidate op is sampled and executed
    heat perturbs vectors
    surprise is measured
    curiosity/reward update rules eventually modify the machine
```

The state vector remains the fetch pointer. There is no separate program
counter unless experiments force that split later.

## V0 Tick

The current C++ core implements this stripped tick:

```text
working_state = normalize(state + input_scale * input)
heat(op_bank)
scores = dot(working_state, op_i)
candidates = top_n(scores)
op = sample_one(candidates)
predicted = normalize(ReLU(working_state + update_scale * op))

observed = normalize(predicted + heat(state))

prediction_error = mean_square(predicted - observed)
curiosity_reward = curiosity_scale * prediction_error
```

Defaults keep heat at zero, so tests and smoke runs are deterministic unless
heat is enabled.

External observation can replace the tick's observed state after the fact:

```text
apply_observation(tick, externally_observed_or_target_state)
```

That recomputes prediction error and curiosity. Curiosity therefore does not
require heat; heat is only one possible source of surprise.

## ICM References

Pathak et al. define curiosity as next-state prediction error in a learned
feature space, with an inverse model used to make the feature space focus on
agent-controllable change rather than raw pixels or uncontrollable noise.

Burda et al. stress-tested curiosity-only learning at scale and found that
prediction-error rewards can work surprisingly well across many environments,
but they also show the key failure mode: stochastic or inherently noisy state can
be attractive forever if surprise is rewarded naively.

For VVM, the useful lesson is not to copy ICM directly. The useful lesson is:

- surprise should be computed in vector state space, not raw pixels
- curiosity should be tied to controllable or learnable change
- heat/noise must not become an infinite reward source
- curiosity should eventually become prediction progress or useful surprise,
  not raw prediction error forever

The attention/rational-curiosity variants are relevant because they try to make
curiosity selective: some dimensions or states should count more than others.
That maps naturally to future VVM masks over state dimensions, op groups, or
actors.

## Universal Training Signals

### Next-State Prediction

The basic self-supervised target is:

```text
predict S(t + horizon) from S(t)
```

Start with horizon 1. Later use wider horizons:

```text
1, 2, 4, 8, 16, ...
```

The open design question is whether each horizon has its own predictor, or
whether the same VVM is stepped internally and compared against delayed state.

### Curiosity

V0 reports:

```text
curiosity_reward = curiosity_scale * prediction_error
```

This is a measured intrinsic reward scalar. In the current code it is dormant:
useful for logging and future experiments, but it does not create behavior by
itself.

For VVM, action selection is op selection:

```text
state/query -> top-k ops -> sampled chosen op
```

So future curiosity should act on op selection pressure, not directly on heat or
input scale:

- make useful surprising chosen paths more likely
- combine with value/return learning over a window
- stay separate from homeostatic gain control

The next version should avoid rewarding uncontrollable heat forever. Candidate
fixes:

- reward prediction improvement instead of raw error
- subtract expected heat error
- keep a running surprise baseline
- only reward error that later becomes predictable
- use attention/masks so only useful dimensions contribute

The target signal is not "maximize loss." It is closer to:

```text
curiosity = novelty_weight * surprise
          + progress_weight * max(previous_error_baseline - current_error, 0)
```

Raw surprise gives the immediate "what was that?" response. Learning progress
keeps the system from getting stuck worshipping noise.

### Credit Assignment

Candidate retrieval is not execution. Each tick:

```text
candidates = top_n(scores)
chosen = sample_one(candidates)
execute(chosen)
```

For the first manual backprop rule, only the chosen op should receive the
gradient/update from that tick. Unchosen candidates were considered but did not
run, so they should not learn from that transition. This keeps the VM
instruction-like and makes truncated backprop simpler.

The current optional routing correction is weak rejection:

```text
if chosen op loss is above threshold:
    move chosen op slightly away from the query
```

This is not backprop through selection. It is a local table-spreading rule so a
bad op stops monopolizing a state region and other nearby ops can bubble up.

### Reinforcement

External reward should enter as a masked scalar stream alongside intrinsic
reward. Missing external reward is not the same as observed zero reward.

```text
reward_total = reward_curiosity
if has_external_reward:
    reward_total += reward_external
```

Reward baselines and normalizers should update only on ticks where
`has_external_reward` is true.

The first RL hook should be a value estimate sampled from state:

```text
V(S_t) -> expected discounted future reward
TD error = r_t + gamma * V(S_t+1) - V(S_t)
```

Later, that TD error can bias op updates, actor selection, or a separate policy
head. For now, the code only reports curiosity reward and does not update ops.

## Heat

Heat is always-on noise in the machine. It is intended to support exploration
and prevent dead vector dynamics.

Current knobs:

- `state_heat_stddev`
- `op_heat_stddev`
- `heat_decay`

The current implementation can heat the active state and the op bank, then
normalizes afterward. This is intentionally simple and probably too crude.

Open questions:

- should op-bank heat be global, local to active ops, or actor-specific?
- should heat decay globally with time, adapt to surprise, or reset per episode?
- should heat be zeroed in stable modes and raised only when curiosity stalls?
- should heat be injected before retrieval, after transition, or both?

The main risk is rattling the machine into meaningless motion. The working
principle for now is: normalize after heat, start with tiny heat, and record
whether surprise decays or stays high.

Track whether curiosity is chasing internal instability:

```text
self_loss_mean
self_loss_slope
activation_mean
chosen_op_entropy
candidate_entropy
op_bank_drift
```

Magnitudes can later be tuned with a separate auto-ISO style homeostatic
controller:

```text
if activation too low: raise input/update/heat scale slightly
if activation too high: lower input/update/heat scale slightly
if self_loss high and flat: lower heat or rejection
if op usage collapses: increase sampling/rejection slightly
```

## Input And Output Sampling

Inputs should be folded into state, not handled by a separate input network at
first:

```text
S = normalize(S + input_scale * input_vector)
```

This keeps the VM pure: the state remains both memory and address.

Outputs should initially be sampled from state:

```text
output = readout(S)
```

For early tasks, readout can start as direct selected dimensions, then become a
small learned linear map. Avoid a large output head early, because it can solve
the task while the VVM core does nothing.

## Actors

The current repo has one actor and one op bank. A likely future architecture is:

```text
many actor states -> shared op bank
```

Questions to test:

- do multiple actors share useful ops?
- does shared heat destroy specialization?
- should actors have private state heat but shared op-bank learning?
- do actors need separate value/curiosity baselines?

## Immediate Tasks

Start with simple continuous sequence tasks:

- copy next scalar/vector
- delayed copy
- alternating bit
- low-period sequence prediction
- noisy sequence with known noise level
- input-response lookup
- small gridworld observation stream

The first useful plot is prediction error over time with different heat values.
