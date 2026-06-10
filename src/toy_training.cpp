#include "vvm/toy_training.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace vvm {
namespace {

void normalize_l2(std::vector<float>& values) {
    const float norm = l2_norm(values);
    if (norm <= 1.0e-8F) {
        const float fill = 1.0F / std::sqrt(static_cast<float>(values.size()));
        std::fill(values.begin(), values.end(), fill);
        return;
    }

    for (float& value : values) {
        value /= norm;
    }
}

std::vector<float> random_nonnegative_unit(std::size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> values(size);
    for (float& value : values) {
        value = dist(rng);
    }
    normalize_l2(values);
    return values;
}

void validate_sample(const ToySample& sample, std::size_t state_dim) {
    if (sample.input.size() != state_dim || sample.target.size() != state_dim) {
        throw std::invalid_argument("toy sample vector size does not match model state_dim");
    }
}

float run_sample_loss(Model& model, const ToySample& sample, const ToyTaskConfig& task_config,
                      std::mt19937& rng, std::size_t clock) {
    std::vector<float> state = neutral_state(model.config().state_dim);
    for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
        (void)model.tick(state, rng, clock + frame, sample.input);
    }
    return Model::prediction_error(state, sample.target);
}

} // namespace

std::vector<float> neutral_state(std::size_t state_dim) {
    if (state_dim == 0U) {
        throw std::invalid_argument("state_dim must be nonzero");
    }

    std::vector<float> state(state_dim, 1.0F / std::sqrt(static_cast<float>(state_dim)));
    return state;
}

ToyDataset make_toy_dataset(const Config& model_config, const ToyTaskConfig& task_config) {
    if (task_config.frames_per_sample == 0U) {
        throw std::invalid_argument("frames_per_sample must be nonzero");
    }
    if (task_config.window_size == 0U) {
        throw std::invalid_argument("window_size must be nonzero");
    }

    std::mt19937 rng(task_config.seed);
    ToyDataset dataset{};
    dataset.train.reserve(task_config.train_samples);
    dataset.test.reserve(task_config.test_samples);

    auto append_samples = [&](std::vector<ToySample>& samples, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            std::vector<float> input = random_nonnegative_unit(model_config.state_dim, rng);
            samples.push_back(ToySample{
                .input = input,
                .target = std::move(input),
            });
        }
    };

    append_samples(dataset.train, task_config.train_samples);
    append_samples(dataset.test, task_config.test_samples);
    return dataset;
}

float evaluate_toy_loss(Model& model, std::span<const ToySample> samples,
                        const ToyTaskConfig& task_config) {
    if (samples.empty()) {
        return 0.0F;
    }

    std::mt19937 rng(task_config.seed ^ 0xBADC0DEU);
    float loss_sum = 0.0F;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        validate_sample(samples[i], model.config().state_dim);
        loss_sum +=
            run_sample_loss(model, samples[i], task_config, rng, i * task_config.frames_per_sample);
    }
    return loss_sum / static_cast<float>(samples.size());
}

LossPoint train_toy_epoch(Model& model, std::span<const ToySample> train_samples,
                          std::span<const ToySample> test_samples, const ToyTaskConfig& task_config,
                          std::size_t epoch) {
    if (train_samples.empty()) {
        return LossPoint{
            .test_loss = evaluate_toy_loss(model, test_samples, task_config),
        };
    }

    TrainConfig train_config{};
    train_config.learning_rate = task_config.learning_rate;
    train_config.recency_decay = task_config.recency_decay;
    train_config.max_grad_norm = task_config.max_grad_norm;
    train_config.rejection_scale = task_config.rejection_scale;
    train_config.rejection_threshold = task_config.rejection_threshold;

    std::mt19937 rng(task_config.seed ^ static_cast<std::uint32_t>(epoch * 0x9E3779B9U));
    std::vector<std::size_t> order(train_samples.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::shuffle(order.begin(), order.end(), rng);

    float train_loss_sum = 0.0F;
    float self_loss_sum = 0.0F;
    std::size_t trained_ticks = 0;
    std::size_t self_ticks = 0;

    for (std::size_t order_index = 0; order_index < order.size(); ++order_index) {
        const ToySample& sample = train_samples[order[order_index]];
        validate_sample(sample, model.config().state_dim);

        std::vector<float> state = neutral_state(model.config().state_dim);
        std::vector<Tick> window;
        window.reserve(task_config.window_size);

        for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
            const std::size_t clock =
                ((epoch * train_samples.size()) + order_index) * task_config.frames_per_sample +
                frame;
            Tick tick = model.tick(state, rng, clock, sample.input);
            tick.observed_state = sample.target;
            tick.prediction_error =
                Model::prediction_error(tick.predicted_state, tick.observed_state);

            train_loss_sum += tick.prediction_error;
            ++trained_ticks;

            if (window.size() == task_config.window_size) {
                window.erase(window.begin());
            }
            window.push_back(std::move(tick));

            (void)model.train_window(window, train_config);
        }

        for (std::size_t frame = 0; frame < task_config.idle_frames_between_samples; ++frame) {
            const std::size_t clock =
                (((epoch * train_samples.size()) + order_index) *
                 (task_config.frames_per_sample + task_config.idle_frames_between_samples)) +
                task_config.frames_per_sample + frame;
            Tick tick = model.tick(state, rng, clock);

            self_loss_sum += tick.prediction_error;
            ++self_ticks;

            if (window.size() == task_config.window_size) {
                window.erase(window.begin());
            }
            window.push_back(std::move(tick));

            (void)model.train_window(window, train_config);
        }
    }

    return LossPoint{
        .train_loss = train_loss_sum / static_cast<float>(trained_ticks),
        .self_loss = self_ticks == 0U ? 0.0F : self_loss_sum / static_cast<float>(self_ticks),
        .test_loss = evaluate_toy_loss(model, test_samples, task_config),
    };
}

} // namespace vvm
