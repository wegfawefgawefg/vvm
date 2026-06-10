#include "vvm/tasks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>

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

std::vector<float> random_signed_unit(std::size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> values(size);
    for (float& value : values) {
        value = dist(rng);
    }
    normalize_l2(values);
    return values;
}

std::vector<float> basis_vector(std::size_t size, std::size_t index) {
    std::vector<float> values(size, 0.0F);
    values[index % size] = 1.0F;
    return values;
}

std::vector<float> bit_vector(std::size_t size, bool bit, VectorRange range) {
    if (range == VectorRange::Signed && size > 1U) {
        std::vector<float> values(size, 0.0F);
        values[0] = bit ? -1.0F : 1.0F;
        values[1] = bit ? 1.0F : -1.0F;
        normalize_l2(values);
        return values;
    }
    return basis_vector(size, bit && size > 1U ? 1U : 0U);
}

std::vector<float> binary_pair_vector(std::size_t size, bool first, bool second,
                                      VectorRange range) {
    std::vector<float> values(size, 0.0F);
    if (range == VectorRange::Signed) {
        values[0] = first ? 1.0F : -1.0F;
        if (size > 1U) {
            values[1] = second ? 1.0F : -1.0F;
        }
        if (size > 2U) {
            values[2] = 1.0F;
        }
        normalize_l2(values);
        return values;
    }

    values[0] = first ? 1.0F : 0.0F;
    if (size > 1U) {
        values[1] = second ? 1.0F : 0.0F;
    }
    if (size > 2U) {
        values[2] = 1.0F;
    }
    normalize_l2(values);
    return values;
}

std::vector<float> sine_phase_vector(std::size_t size, float phase, VectorRange range) {
    constexpr float kPi = 3.14159265358979323846F;
    std::vector<float> values(size, 0.0F);
    const auto encode = [range](float value) {
        return range == VectorRange::Signed ? value : (value + 1.0F) * 0.5F;
    };
    values[0] = encode(std::sin(phase));
    if (size > 1U) {
        values[1] = encode(std::cos(phase));
    }
    if (size > 2U) {
        values[2] = encode(std::sin(2.0F * phase));
    }
    if (size > 3U) {
        values[3] = encode(std::cos(2.0F * phase));
    }
    if (size > 4U) {
        values[4] = encode(std::sin(phase + kPi * 0.25F));
    }
    normalize_l2(values);
    return values;
}

TaskSample make_sample(TaskKind task, const Config& model_config, std::size_t index,
                       std::size_t offset, VectorRange range, std::mt19937& rng) {
    const std::size_t state_dim = model_config.state_dim;
    switch (task) {
    case TaskKind::CopyInput: {
        std::vector<float> input = range == VectorRange::Signed
                                       ? random_signed_unit(state_dim, rng)
                                       : random_nonnegative_unit(state_dim, rng);
        return TaskSample{
            .input = input,
            .target = std::move(input),
        };
    }
    case TaskKind::DelayedCopy: {
        std::vector<float> input = range == VectorRange::Signed
                                       ? random_signed_unit(state_dim, rng)
                                       : random_nonnegative_unit(state_dim, rng);
        return TaskSample{
            .input = input,
            .target = std::move(input),
        };
    }
    case TaskKind::AlternatingBit: {
        const bool bit = ((index + offset) % 2U) != 0U;
        return TaskSample{
            .input = bit_vector(state_dim, bit, range),
            .target = bit_vector(state_dim, !bit, range),
        };
    }
    case TaskKind::Xor: {
        const std::size_t pattern = (index + offset) % 4U;
        const bool first = (pattern & 0x1U) != 0U;
        const bool second = (pattern & 0x2U) != 0U;
        return TaskSample{
            .input = binary_pair_vector(state_dim, first, second, range),
            .target = bit_vector(state_dim, first != second, range),
        };
    }
    case TaskKind::SineNext: {
        constexpr float kPi = 3.14159265358979323846F;
        constexpr std::size_t kPeriod = 32;
        const float phase =
            2.0F * kPi *
            (static_cast<float>((index + offset) % kPeriod) / static_cast<float>(kPeriod));
        const float next_phase = phase + (2.0F * kPi / static_cast<float>(kPeriod));
        return TaskSample{
            .input = sine_phase_vector(state_dim, phase, range),
            .target = sine_phase_vector(state_dim, next_phase, range),
        };
    }
    }
    throw std::invalid_argument("unknown task kind");
}

void validate_sample(const TaskSample& sample, std::size_t state_dim) {
    if (sample.input.size() != state_dim || sample.target.size() != state_dim) {
        throw std::invalid_argument("task sample vector size does not match model state_dim");
    }
}

float run_sample_loss(Model& model, const TaskSample& sample, const TaskConfig& task_config,
                      std::mt19937& rng, std::size_t clock) {
    std::vector<float> state = neutral_state(model.config().state_dim);
    for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
        (void)model.tick(state, rng, clock + frame, sample.input);
    }
    if (task_config.task == TaskKind::DelayedCopy) {
        for (std::size_t frame = 0; frame < task_config.idle_frames_between_samples; ++frame) {
            (void)model.tick(state, rng, clock + task_config.frames_per_sample + frame);
        }
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

TaskDataset make_task_dataset(const Config& model_config, const TaskConfig& task_config) {
    if (task_config.frames_per_sample == 0U) {
        throw std::invalid_argument("frames_per_sample must be nonzero");
    }
    if (task_config.window_size == 0U) {
        throw std::invalid_argument("window_size must be nonzero");
    }
    if (model_config.state_dim == 0U) {
        throw std::invalid_argument("state_dim must be nonzero");
    }

    std::mt19937 rng(task_config.seed);
    TaskDataset dataset{};
    dataset.train.reserve(task_config.train_samples);
    dataset.test.reserve(task_config.test_samples);

    auto append_samples = [&](std::vector<TaskSample>& samples, std::size_t count,
                              std::size_t offset) {
        for (std::size_t i = 0; i < count; ++i) {
            samples.push_back(make_sample(task_config.task, model_config, i, offset,
                                          task_config.vector_range, rng));
        }
    };

    append_samples(dataset.train, task_config.train_samples, 0U);
    append_samples(dataset.test, task_config.test_samples, task_config.train_samples);
    return dataset;
}

const char* task_name(TaskKind task) {
    switch (task) {
    case TaskKind::CopyInput:
        return "copy-input";
    case TaskKind::DelayedCopy:
        return "delayed-copy";
    case TaskKind::AlternatingBit:
        return "alternating-bit";
    case TaskKind::Xor:
        return "xor";
    case TaskKind::SineNext:
        return "sine-next";
    }
    return "unknown";
}

float evaluate_task_loss(Model& model, std::span<const TaskSample> samples,
                         const TaskConfig& task_config) {
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

LossPoint train_task_epoch(Model& model, std::span<const TaskSample> train_samples,
                           std::span<const TaskSample> test_samples, const TaskConfig& task_config,
                           std::size_t epoch) {
    if (train_samples.empty()) {
        return LossPoint{
            .test_loss = evaluate_task_loss(model, test_samples, task_config),
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
        const TaskSample& sample = train_samples[order[order_index]];
        validate_sample(sample, model.config().state_dim);

        std::vector<float> state = neutral_state(model.config().state_dim);
        std::vector<Tick> window;
        window.reserve(task_config.window_size);

        for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
            const std::size_t clock =
                ((epoch * train_samples.size()) + order_index) * task_config.frames_per_sample +
                frame;
            Tick tick = model.tick(state, rng, clock, sample.input);
            apply_observation(tick, sample.target, model.config().curiosity_scale);

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
            if (task_config.task == TaskKind::DelayedCopy) {
                apply_observation(tick, sample.target, model.config().curiosity_scale);
            }

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
        .test_loss = evaluate_task_loss(model, test_samples, task_config),
    };
}

} // namespace vvm
