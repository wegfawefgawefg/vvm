#include "vvm/model.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

void test_dot_product() {
    const float a_values[] = {1.0F, 0.0F};
    const float b_values[] = {0.0F, 1.0F};
    const float c_values[] = {2.0F, 0.0F};

    assert(std::fabs(vvm::dot_product(a_values, b_values)) < 1.0e-6F);
    assert(std::fabs(vvm::dot_product(a_values, c_values) - 2.0F) < 1.0e-6F);
}

void test_run_shape() {
    vvm::Config config{};
    config.state_dim = 16;
    config.num_ops = 32;
    config.candidate_count = 4;
    config.steps = 3;

    vvm::Model model(config);
    const std::vector<float> initial_state = model.seeded_state();
    const vvm::RunResult result = model.run(initial_state);

    assert(result.state.size() == config.state_dim);
    assert(result.trace.size() == config.steps);

    for (const vvm::StepTrace& trace : result.trace) {
        assert(trace.retrieval.candidate_indices.size() == config.candidate_count);
        assert(trace.retrieval.candidate_weights.size() == config.candidate_count);
        assert(trace.retrieval.chosen_index < config.num_ops);
        assert(std::find(trace.retrieval.candidate_indices.begin(),
                         trace.retrieval.candidate_indices.end(),
                         trace.retrieval.chosen_index) != trace.retrieval.candidate_indices.end());
        assert(trace.state_norm > 0.0F);
        assert(trace.activation_mean >= 0.0F);
        assert(trace.prediction_error >= 0.0F);
        assert(trace.curiosity_reward >= 0.0F);
    }
}

void test_prediction_error() {
    const float predicted[] = {1.0F, 0.0F};
    const float observed[] = {0.0F, 1.0F};
    assert(std::fabs(vvm::Model::prediction_error(predicted, observed) - 1.0F) < 1.0e-6F);
}

void test_heat_creates_curiosity() {
    vvm::Config config{};
    config.state_dim = 16;
    config.num_ops = 32;
    config.candidate_count = 4;
    config.steps = 3;
    config.state_heat_stddev = 0.1F;

    vvm::Model model(config);
    const std::vector<float> initial_state = model.seeded_state();
    const vvm::RunResult result = model.run(initial_state);

    bool saw_error = false;
    for (const vvm::StepTrace& trace : result.trace) {
        saw_error = saw_error || trace.prediction_error > 0.0F;
        assert(trace.state_heat_stddev > 0.0F);
    }
    assert(saw_error);
}

void test_tick_masks_missing_external_reward() {
    vvm::Config config{};
    config.state_dim = 8;
    config.num_ops = 16;
    config.candidate_count = 4;
    config.curiosity_scale = 2.0F;
    config.state_heat_stddev = 0.05F;

    vvm::Model model(config);
    std::vector<float> state = model.seeded_state();
    std::mt19937 rng(config.seed);

    const vvm::Tick no_external = model.tick(state, rng, 0);
    assert(!no_external.reward.has_external_reward);
    assert(std::fabs(no_external.reward.total_reward - no_external.reward.curiosity_reward) <
           1.0e-6F);

    vvm::RewardSignal reward{};
    reward.external_reward = 3.0F;
    reward.has_external_reward = true;

    const vvm::Tick with_external = model.tick(state, rng, 1, {}, reward);
    assert(with_external.reward.has_external_reward);
    assert(std::fabs(with_external.reward.total_reward -
                     (with_external.reward.curiosity_reward + reward.external_reward)) < 1.0e-6F);
}

void test_train_window_updates_only_chosen_ops() {
    vvm::Config config{};
    config.state_dim = 8;
    config.num_ops = 16;
    config.candidate_count = 4;
    config.update_scale = 1.0F;

    vvm::Model model(config);
    std::vector<float> state = model.seeded_state();
    std::mt19937 rng(config.seed);
    vvm::Tick tick = model.tick(state, rng, 0);

    std::size_t active_index = 0;
    for (std::size_t i = 0; i < tick.post_relu.size(); ++i) {
        if (tick.post_relu[i] > 0.0F) {
            active_index = i;
            break;
        }
    }

    tick.observed_state = tick.predicted_state;
    tick.observed_state[active_index] += 0.5F;
    const float target_norm = vvm::l2_norm(tick.observed_state);
    for (float& value : tick.observed_state) {
        value /= target_norm;
    }
    tick.prediction_error = vvm::Model::prediction_error(tick.predicted_state, tick.observed_state);

    const std::vector<float> before(model.op_bank().begin(), model.op_bank().end());

    vvm::TrainConfig train_config{};
    train_config.learning_rate = 0.5F;
    train_config.max_grad_norm = 10.0F;
    const vvm::TrainResult result =
        model.train_window(std::span<const vvm::Tick>(&tick, 1), train_config);

    assert(result.tick_count == 1U);
    assert(result.updated_ops == 1U);
    assert(result.loss > 0.0F);

    const std::span<const float> after = model.op_bank();
    std::size_t changed_ops = 0;
    for (std::size_t op = 0; op < config.num_ops; ++op) {
        bool changed = false;
        for (std::size_t i = 0; i < config.state_dim; ++i) {
            const std::size_t offset = (op * config.state_dim) + i;
            changed = changed || std::fabs(before[offset] - after[offset]) > 1.0e-6F;
        }
        if (changed) {
            ++changed_ops;
            assert(op == tick.chosen_op);
            assert(std::fabs(vvm::l2_norm(after.subspan(op * config.state_dim, config.state_dim)) -
                             1.0F) < 1.0e-5F);
        }
    }
    assert(changed_ops == 1U);
}

void test_invalid_config() {
    vvm::Config config{};
    config.num_ops = 2;
    config.candidate_count = 3;

    bool threw = false;
    try {
        const vvm::Model model(config);
        (void)model;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_dot_product();
    test_run_shape();
    test_prediction_error();
    test_heat_creates_curiosity();
    test_tick_masks_missing_external_reward();
    test_train_window_updates_only_chosen_ops();
    test_invalid_config();

    std::cout << "vvm core tests passed\n";
    return 0;
}
