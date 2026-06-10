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

This is only a diagnostic and toy reward. The next version should avoid rewarding
uncontrollable heat forever. Candidate fixes:

- reward prediction improvement instead of raw error
- subtract expected heat error
- keep a running surprise baseline
- only reward error that later becomes predictable
- use attention/masks so only useful dimensions contribute

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

### Reinforcement

External reward should enter as a scalar stream alongside intrinsic reward:

```text
reward_total = reward_external + reward_curiosity
```

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

For toy problems, readout can start as direct selected dimensions, then become a
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

## Immediate Toy Problems

Start with simple continuous sequence tasks:

- copy next scalar/vector
- delayed copy
- alternating bit
- low-period sequence prediction
- noisy sequence with known noise level
- input-response lookup
- small gridworld observation stream

The first useful plot is prediction error over time with different heat values.
