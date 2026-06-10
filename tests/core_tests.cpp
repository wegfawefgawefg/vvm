#include "vvm/model.hpp"
#include "vvm/tasks.hpp"

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
    assert(model.parameter_count() == config.state_dim * config.num_ops);
    assert(model.parameter_bytes() == model.parameter_count() * sizeof(float));
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

void test_activation_modes_tick() {
    const vvm::ActivationKind activations[] = {
        vvm::ActivationKind::Relu,
        vvm::ActivationKind::LeakyRelu,
        vvm::ActivationKind::Clamp,
        vvm::ActivationKind::Deadzone,
    };

    for (const vvm::ActivationKind activation : activations) {
        vvm::Config config{};
        config.state_dim = 8;
        config.num_ops = 16;
        config.candidate_count = 4;
        config.activation = activation;

        vvm::Model model(config);
        std::vector<float> state = model.seeded_state();
        std::mt19937 rng(config.seed);
        const vvm::Tick tick = model.tick(state, rng, 0);

        assert(tick.pre_activation.size() == config.state_dim);
        assert(tick.post_activation.size() == config.state_dim);
        assert(std::fabs(vvm::l2_norm(tick.predicted_state) - 1.0F) < 1.0e-5F);
    }
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
    for (std::size_t i = 0; i < tick.post_activation.size(); ++i) {
        if (std::fabs(tick.post_activation[i]) > 0.0F) {
            active_index = i;
            break;
        }
    }

    std::vector<float> target = tick.predicted_state;
    target[active_index] += 0.5F;
    const float target_norm = vvm::l2_norm(target);
    for (float& value : target) {
        value /= target_norm;
    }
    vvm::apply_observation(tick, target, config.curiosity_scale);

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

void test_train_window_moves_prediction_toward_observation() {
    vvm::Config config{};
    config.state_dim = 8;
    config.num_ops = 16;
    config.candidate_count = 4;
    config.update_scale = 1.0F;
    config.sample_retrieval = false;
    config.state_heat_stddev = 0.0F;
    config.op_heat_stddev = 0.0F;

    vvm::Model model(config);
    std::vector<float> state = model.seeded_state();
    std::mt19937 rng(config.seed);
    vvm::Tick tick = model.tick(state, rng, 0);

    std::size_t active_index = 0;
    for (std::size_t i = 0; i < tick.post_activation.size(); ++i) {
        if (std::fabs(tick.post_activation[i]) > 0.0F) {
            active_index = i;
            break;
        }
    }

    std::vector<float> target = tick.predicted_state;
    target[active_index] += 0.25F;
    const float target_norm = vvm::l2_norm(target);
    for (float& value : target) {
        value /= target_norm;
    }
    vvm::apply_observation(tick, target, config.curiosity_scale);
    const float before_loss = vvm::Model::prediction_error(tick.predicted_state, target);

    vvm::TrainConfig train_config{};
    train_config.learning_rate = 0.1F;
    train_config.max_grad_norm = 10.0F;
    const vvm::TrainResult result =
        model.train_window(std::span<const vvm::Tick>(&tick, 1), train_config);
    assert(result.updated_ops == 1U);

    const std::vector<float> after_prediction = model.predict_next(tick.working_state);
    const float after_loss = vvm::Model::prediction_error(after_prediction, target);

    assert(after_loss < before_loss);
}

void test_rejection_lowers_bad_op_affinity() {
    vvm::Config config{};
    config.state_dim = 8;
    config.num_ops = 16;
    config.candidate_count = 4;
    config.update_scale = 1.0F;

    vvm::Model model(config);
    std::vector<float> state = model.seeded_state();
    std::mt19937 rng(config.seed);
    vvm::Tick tick = model.tick(state, rng, 0);

    tick.observed_state = tick.predicted_state;
    tick.prediction_error = 1.0F;

    const std::span<const float> before = model.op_bank();
    const std::span<const float> before_op =
        before.subspan(tick.chosen_op * config.state_dim, config.state_dim);
    const float before_affinity = vvm::dot_product(before_op, tick.working_state);

    vvm::TrainConfig train_config{};
    train_config.learning_rate = 0.5F;
    train_config.max_grad_norm = 10.0F;
    train_config.rejection_scale = 1.0F;
    train_config.rejection_threshold = 0.0F;
    (void)model.train_window(std::span<const vvm::Tick>(&tick, 1), train_config);

    const std::span<const float> after = model.op_bank();
    const std::span<const float> after_op =
        after.subspan(tick.chosen_op * config.state_dim, config.state_dim);
    const float after_affinity = vvm::dot_product(after_op, tick.working_state);

    assert(after_affinity < before_affinity);
}

void test_observation_creates_curiosity_without_heat() {
    vvm::Config config{};
    config.state_dim = 8;
    config.num_ops = 16;
    config.candidate_count = 4;
    config.curiosity_scale = 2.0F;
    config.state_heat_stddev = 0.0F;
    config.op_heat_stddev = 0.0F;

    vvm::Model model(config);
    std::vector<float> state = model.seeded_state();
    std::mt19937 rng(config.seed);
    vvm::Tick tick = model.tick(state, rng, 0);
    assert(tick.prediction_error == 0.0F);
    assert(tick.reward.curiosity_reward == 0.0F);

    std::vector<float> observed = tick.predicted_state;
    observed[0] += 0.75F;
    const float observed_norm = vvm::l2_norm(observed);
    for (float& value : observed) {
        value /= observed_norm;
    }

    vvm::apply_observation(tick, observed, config.curiosity_scale);
    assert(tick.prediction_error > 0.0F);
    assert(std::fabs(tick.reward.curiosity_reward -
                     (config.curiosity_scale * tick.prediction_error)) < 1.0e-6F);
    assert(std::fabs(tick.reward.total_reward - tick.reward.curiosity_reward) < 1.0e-6F);
}

void test_task_training_runs(vvm::TaskKind task_kind, std::size_t idle_frames) {
    vvm::Config config{};
    config.state_dim = 8;
    config.num_ops = 32;
    config.candidate_count = 4;
    config.state_heat_stddev = 0.0F;
    config.op_heat_stddev = 0.0F;

    vvm::TaskConfig task_config{};
    task_config.task = task_kind;
    task_config.train_samples = 8;
    task_config.test_samples = 4;
    task_config.frames_per_sample = 4;
    task_config.idle_frames_between_samples = idle_frames;
    task_config.window_size = 4;
    task_config.learning_rate = 0.05F;

    vvm::Model model(config);
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);
    const float before = vvm::evaluate_task_loss(model, dataset.test, task_config);
    const vvm::LossPoint loss =
        vvm::train_task_epoch(model, dataset.train, dataset.test, task_config, 0);

    assert(std::isfinite(before));
    assert(std::isfinite(loss.train_loss));
    assert(std::isfinite(loss.test_loss));
    assert(loss.train_loss >= 0.0F);
    assert(loss.test_loss >= 0.0F);
    if (idle_frames > 0U) {
        assert(std::isfinite(loss.self_loss));
        assert(loss.self_loss >= 0.0F);
    }
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
    test_activation_modes_tick();
    test_heat_creates_curiosity();
    test_tick_masks_missing_external_reward();
    test_train_window_updates_only_chosen_ops();
    test_train_window_moves_prediction_toward_observation();
    test_rejection_lowers_bad_op_affinity();
    test_observation_creates_curiosity_without_heat();
    test_task_training_runs(vvm::TaskKind::CopyInput, 0U);
    test_task_training_runs(vvm::TaskKind::DelayedCopy, 4U);
    test_task_training_runs(vvm::TaskKind::AlternatingBit, 0U);
    test_task_training_runs(vvm::TaskKind::Xor, 0U);
    test_task_training_runs(vvm::TaskKind::SineNext, 0U);
    test_invalid_config();

    std::cout << "vvm core tests passed\n";
    return 0;
}
