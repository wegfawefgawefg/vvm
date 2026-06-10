#include "vvm/tasks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
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

std::vector<float> class_vector(std::size_t size, int label, int class_count, VectorRange range) {
    if (label < 0 || class_count <= 0 || label >= class_count) {
        throw std::invalid_argument("invalid class target");
    }

    std::vector<float> values(size, 0.0F);
    if (range == VectorRange::Signed) {
        const float off_value =
            class_count <= 2 ? -1.0F : -2.0F / static_cast<float>(class_count - 2);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<int>(i % static_cast<std::size_t>(class_count)) == label
                            ? 1.0F
                            : off_value;
        }
    } else {
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] =
                static_cast<int>(i % static_cast<std::size_t>(class_count)) == label ? 1.0F : 0.0F;
        }
    }
    normalize_l2(values);
    return values;
}

std::vector<float> uniform_weights(std::size_t size, float value = 1.0F) {
    return std::vector<float>(size, value);
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

std::uint32_t read_be_u32(std::istream& stream, const std::filesystem::path& path) {
    unsigned char bytes[4] = {};
    stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!stream) {
        throw std::runtime_error("failed to read IDX header from " + path.string());
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

std::ifstream open_binary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open " + path.string());
    }
    return stream;
}

float class_loss_weight_or(float configured, float fallback) {
    return configured > 0.0F ? configured : fallback;
}

std::size_t class_register_count(const TaskConfig& task_config, std::size_t class_count) {
    return task_config.class_registers == 0U ? class_count : task_config.class_registers;
}

TaskSample make_mnist_sample(std::span<const unsigned char, 784> pixels, unsigned char label,
                             std::size_t state_dim, const TaskConfig& task_config) {
    constexpr std::size_t kImageDims = 784;
    constexpr std::size_t kDigitOffset = 784;
    constexpr std::size_t kDigitClasses = 10;
    const std::size_t class_dims = class_register_count(task_config, kDigitClasses);
    if (class_dims < kDigitClasses) {
        throw std::invalid_argument("mnist class_registers must be >= 10");
    }
    if (state_dim < kImageDims + class_dims) {
        throw std::invalid_argument("mnist task requires state_dim >= 784 + class_registers");
    }

    std::vector<float> input(state_dim, 0.0F);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        input[i] = static_cast<float>(pixels[i]) / 255.0F;
    }
    normalize_l2(input);

    std::vector<float> target(state_dim, 0.0F);
    std::vector<float> target_weights = uniform_weights(state_dim, task_config.world_loss_weight);
    const float image_weight = task_config.world_loss_weight > 0.0F ? 1.0F : 0.0F;
    const float class_value_scale = task_config.class_value_scale;
    const float class_target_weight = class_loss_weight_or(task_config.class_loss_weight, 64.0F);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        target[i] = image_weight * input[i];
    }
    const std::vector<float> digit =
        class_vector(class_dims, static_cast<int>(label), static_cast<int>(kDigitClasses),
                     task_config.vector_range);
    for (std::size_t i = 0; i < digit.size(); ++i) {
        target[kDigitOffset + i] = class_value_scale * digit[i];
        target_weights[kDigitOffset + i] = class_target_weight;
    }
    normalize_l2(target);

    return TaskSample{
        .input = std::move(input),
        .target = std::move(target),
        .target_weights = std::move(target_weights),
        .label = static_cast<int>(label),
        .class_count = static_cast<int>(kDigitClasses),
        .class_dims = class_dims,
        .class_offset = kDigitOffset,
    };
}

TaskSample make_mnist_binary_sample(std::span<const unsigned char, 784> pixels, unsigned char label,
                                    std::size_t state_dim, const TaskConfig& task_config) {
    constexpr std::size_t kImageDims = 784;
    constexpr std::size_t kDigitOffset = 784;
    constexpr std::size_t kDigitClasses = 2;
    const std::size_t class_dims = class_register_count(task_config, kDigitClasses);
    if (class_dims < kDigitClasses) {
        throw std::invalid_argument("mnist-01 class_registers must be >= 2");
    }
    if (state_dim < kImageDims + class_dims) {
        throw std::invalid_argument("mnist-01 task requires state_dim >= 784 + class_registers");
    }
    if (label > 1U) {
        throw std::invalid_argument("mnist-01 sample requires label 0 or 1");
    }

    std::vector<float> input(state_dim, 0.0F);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        input[i] = static_cast<float>(pixels[i]) / 255.0F;
    }
    normalize_l2(input);

    std::vector<float> target(state_dim, 0.0F);
    std::vector<float> target_weights = uniform_weights(state_dim, task_config.world_loss_weight);
    const float image_weight = task_config.world_loss_weight > 0.0F ? 1.0F : 0.0F;
    const float class_value_scale = task_config.class_value_scale;
    const float class_target_weight = class_loss_weight_or(task_config.class_loss_weight, 128.0F);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        target[i] = image_weight * input[i];
    }
    const std::vector<float> digit =
        class_vector(class_dims, static_cast<int>(label), static_cast<int>(kDigitClasses),
                     task_config.vector_range);
    for (std::size_t i = 0; i < digit.size(); ++i) {
        target[kDigitOffset + i] = class_value_scale * digit[i];
        target_weights[kDigitOffset + i] = class_target_weight;
    }
    normalize_l2(target);

    return TaskSample{
        .input = std::move(input),
        .target = std::move(target),
        .target_weights = std::move(target_weights),
        .label = static_cast<int>(label),
        .class_count = static_cast<int>(kDigitClasses),
        .class_dims = class_dims,
        .class_offset = kDigitOffset,
    };
}

void append_mnist_split(std::vector<TaskSample>& samples, const std::filesystem::path& images_path,
                        const std::filesystem::path& labels_path, std::size_t requested_count,
                        std::size_t state_dim, const TaskConfig& task_config) {
    std::ifstream images = open_binary(images_path);
    std::ifstream labels = open_binary(labels_path);

    const std::uint32_t image_magic = read_be_u32(images, images_path);
    const std::uint32_t image_count = read_be_u32(images, images_path);
    const std::uint32_t rows = read_be_u32(images, images_path);
    const std::uint32_t cols = read_be_u32(images, images_path);

    const std::uint32_t label_magic = read_be_u32(labels, labels_path);
    const std::uint32_t label_count = read_be_u32(labels, labels_path);

    if (image_magic != 2051U || label_magic != 2049U || rows != 28U || cols != 28U ||
        image_count != label_count) {
        throw std::runtime_error("invalid MNIST IDX files in " +
                                 images_path.parent_path().string());
    }

    const std::size_t count =
        std::min<std::size_t>(requested_count, static_cast<std::size_t>(image_count));
    std::array<unsigned char, 784> pixels = {};
    for (std::size_t i = 0; i < count; ++i) {
        unsigned char label = 0;
        images.read(reinterpret_cast<char*>(pixels.data()),
                    static_cast<std::streamsize>(pixels.size()));
        labels.read(reinterpret_cast<char*>(&label), 1);
        if (!images || !labels) {
            throw std::runtime_error("truncated MNIST IDX files in " +
                                     images_path.parent_path().string());
        }
        if (label >= 10U) {
            throw std::runtime_error("invalid MNIST label in " + labels_path.string());
        }
        samples.push_back(make_mnist_sample(pixels, label, state_dim, task_config));
    }
}

void append_mnist_binary_split(std::vector<TaskSample>& samples,
                               const std::filesystem::path& images_path,
                               const std::filesystem::path& labels_path,
                               std::size_t requested_count, std::size_t state_dim,
                               const TaskConfig& task_config) {
    std::ifstream images = open_binary(images_path);
    std::ifstream labels = open_binary(labels_path);

    const std::uint32_t image_magic = read_be_u32(images, images_path);
    const std::uint32_t image_count = read_be_u32(images, images_path);
    const std::uint32_t rows = read_be_u32(images, images_path);
    const std::uint32_t cols = read_be_u32(images, images_path);

    const std::uint32_t label_magic = read_be_u32(labels, labels_path);
    const std::uint32_t label_count = read_be_u32(labels, labels_path);

    if (image_magic != 2051U || label_magic != 2049U || rows != 28U || cols != 28U ||
        image_count != label_count) {
        throw std::runtime_error("invalid MNIST IDX files in " +
                                 images_path.parent_path().string());
    }

    const std::array<std::size_t, 2> target_counts = {
        requested_count / 2U,
        requested_count - (requested_count / 2U),
    };
    std::array<std::size_t, 2> seen_counts = {};
    std::array<unsigned char, 784> pixels = {};
    for (std::size_t i = 0; i < image_count && (seen_counts[0] < target_counts[0] ||
                                                seen_counts[1] < target_counts[1]);
         ++i) {
        unsigned char label = 0;
        images.read(reinterpret_cast<char*>(pixels.data()),
                    static_cast<std::streamsize>(pixels.size()));
        labels.read(reinterpret_cast<char*>(&label), 1);
        if (!images || !labels) {
            throw std::runtime_error("truncated MNIST IDX files in " +
                                     images_path.parent_path().string());
        }
        if (label <= 1U && seen_counts[static_cast<std::size_t>(label)] <
                               target_counts[static_cast<std::size_t>(label)]) {
            samples.push_back(make_mnist_binary_sample(pixels, label, state_dim, task_config));
            ++seen_counts[static_cast<std::size_t>(label)];
        }
    }
    if (seen_counts[0] != target_counts[0] || seen_counts[1] != target_counts[1]) {
        throw std::runtime_error("not enough MNIST 0/1 samples in " +
                                 images_path.parent_path().string());
    }
}

TaskSample make_sample(TaskKind task, const Config& model_config, std::size_t index,
                       std::size_t offset, const TaskConfig& task_config, std::mt19937& rng) {
    const std::size_t state_dim = model_config.state_dim;
    const VectorRange range = task_config.vector_range;
    switch (task) {
    case TaskKind::CopyInput: {
        std::vector<float> input = range == VectorRange::Signed
                                       ? random_signed_unit(state_dim, rng)
                                       : random_nonnegative_unit(state_dim, rng);
        return TaskSample{
            .input = input,
            .target = std::move(input),
            .target_weights = {},
        };
    }
    case TaskKind::DelayedCopy: {
        std::vector<float> input = range == VectorRange::Signed
                                       ? random_signed_unit(state_dim, rng)
                                       : random_nonnegative_unit(state_dim, rng);
        return TaskSample{
            .input = input,
            .target = std::move(input),
            .target_weights = {},
        };
    }
    case TaskKind::Linear2: {
        std::normal_distribution<float> noise(0.0F, 0.08F);
        const bool label = ((index + offset) % 2U) != 0U;
        std::vector<float> input(state_dim, 0.0F);
        const std::size_t class_offset = state_dim > 2U ? state_dim - 2U : 0U;
        input[0] = label ? 1.0F : -1.0F;
        for (std::size_t i = 1; i < class_offset; ++i) {
            input[i] = noise(rng);
        }
        if (range == VectorRange::Nonnegative) {
            for (std::size_t i = 0; i < class_offset; ++i) {
                input[i] = (input[i] + 1.0F) * 0.5F;
            }
        }
        normalize_l2(input);
        std::vector<float> target = input;
        std::vector<float> target_weights = uniform_weights(state_dim, 1.0F);
        const std::vector<float> class_target = class_vector(2, label ? 1 : 0, 2, range);
        const float class_value_scale = task_config.class_value_scale;
        const float class_target_weight =
            class_loss_weight_or(task_config.class_loss_weight, 16.0F);
        for (std::size_t i = 0; i < class_target.size(); ++i) {
            target[class_offset + i] = class_value_scale * class_target[i];
            target_weights[class_offset + i] = class_target_weight;
        }
        normalize_l2(target);
        return TaskSample{
            .input = std::move(input),
            .target = std::move(target),
            .target_weights = std::move(target_weights),
            .label = label ? 1 : 0,
            .class_count = 2,
            .class_dims = class_target.size(),
            .class_offset = class_offset,
        };
    }
    case TaskKind::Basis4: {
        const int label = static_cast<int>((index + offset) % 4U);
        return TaskSample{
            .input = basis_vector(state_dim, static_cast<std::size_t>(label)),
            .target = class_vector(state_dim, label, 4, range),
            .target_weights = {},
            .label = label,
            .class_count = 4,
            .class_dims = state_dim,
        };
    }
    case TaskKind::AlternatingBit: {
        const bool bit = ((index + offset) % 2U) != 0U;
        return TaskSample{
            .input = bit_vector(state_dim, bit, range),
            .target = bit_vector(state_dim, !bit, range),
            .target_weights = {},
            .label = !bit ? 1 : 0,
            .class_count = 2,
            .class_dims = state_dim,
        };
    }
    case TaskKind::Xor: {
        const std::size_t pattern = (index + offset) % 4U;
        const bool first = (pattern & 0x1U) != 0U;
        const bool second = (pattern & 0x2U) != 0U;
        return TaskSample{
            .input = binary_pair_vector(state_dim, first, second, range),
            .target = bit_vector(state_dim, first != second, range),
            .target_weights = {},
            .label = first != second ? 1 : 0,
            .class_count = 2,
            .class_dims = state_dim,
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
            .target_weights = {},
        };
    }
    case TaskKind::Mnist:
    case TaskKind::Mnist01:
        break;
    }
    throw std::invalid_argument("unknown task kind");
}

TaskDataset make_mnist_dataset(const Config& model_config, const TaskConfig& task_config) {
    if (model_config.state_dim < 794U) {
        throw std::invalid_argument("mnist task requires --state-dim 794 or larger");
    }

    const std::filesystem::path root(task_config.mnist_dir);
    TaskDataset dataset{};
    dataset.train.reserve(task_config.train_samples);
    dataset.test.reserve(task_config.test_samples);

    append_mnist_split(dataset.train, root / "train-images-idx3-ubyte",
                       root / "train-labels-idx1-ubyte", task_config.train_samples,
                       model_config.state_dim, task_config);
    append_mnist_split(dataset.test, root / "t10k-images-idx3-ubyte",
                       root / "t10k-labels-idx1-ubyte", task_config.test_samples,
                       model_config.state_dim, task_config);
    return dataset;
}

TaskDataset make_mnist_binary_dataset(const Config& model_config, const TaskConfig& task_config) {
    if (model_config.state_dim < 786U) {
        throw std::invalid_argument("mnist-01 task requires --state-dim 786 or larger");
    }

    const std::filesystem::path root(task_config.mnist_dir);
    TaskDataset dataset{};
    dataset.train.reserve(task_config.train_samples);
    dataset.test.reserve(task_config.test_samples);

    append_mnist_binary_split(dataset.train, root / "train-images-idx3-ubyte",
                              root / "train-labels-idx1-ubyte", task_config.train_samples,
                              model_config.state_dim, task_config);
    append_mnist_binary_split(dataset.test, root / "t10k-images-idx3-ubyte",
                              root / "t10k-labels-idx1-ubyte", task_config.test_samples,
                              model_config.state_dim, task_config);
    return dataset;
}

void validate_sample(const TaskSample& sample, std::size_t state_dim) {
    if (sample.input.size() != state_dim || sample.target.size() != state_dim) {
        throw std::invalid_argument("task sample vector size does not match model state_dim");
    }
    if (!sample.target_weights.empty() && sample.target_weights.size() != state_dim) {
        throw std::invalid_argument(
            "task sample target weights size does not match model state_dim");
    }
    if (sample.class_count > 0 && sample.class_offset + sample.class_dims > state_dim) {
        throw std::invalid_argument("task sample class range exceeds model state_dim");
    }
}

float class_score(std::span<const float> state, int class_count, std::size_t class_dims,
                  std::size_t class_offset, VectorRange range, int candidate) {
    if (class_count <= 0 || class_dims == 0U || class_offset + class_dims > state.size()) {
        throw std::invalid_argument("invalid class count");
    }
    if (candidate < 0 || candidate >= class_count) {
        throw std::invalid_argument("invalid class candidate");
    }

    float score = 0.0F;
    for (std::size_t i = 0; i < class_dims; ++i) {
        const bool matches =
            static_cast<int>(i % static_cast<std::size_t>(class_count)) == candidate;
        const float value = state[class_offset + i];
        if (range == VectorRange::Signed) {
            const float off_value =
                class_count <= 2 ? -1.0F : -2.0F / static_cast<float>(class_count - 2);
            score += value * (matches ? 1.0F : off_value);
        } else if (matches) {
            score += value;
        }
    }
    return score;
}

int predicted_class(std::span<const float> state, int class_count, std::size_t class_dims,
                    std::size_t class_offset, VectorRange range) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int candidate = 0; candidate < class_count; ++candidate) {
        const float score =
            class_score(state, class_count, class_dims, class_offset, range, candidate);
        if (score > best_score) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

void record_tick_diagnostics(const Tick& tick, const TaskSample& sample, std::size_t num_ops,
                             LossPoint& loss) {
    loss.state_heat_l2 += tick.state_heat_l2;
    loss.op_heat_l2 += tick.op_heat_l2;
    if (!tick.op_heat_l2_by_op.empty()) {
        if (loss.op_heat_l2_by_op.empty()) {
            loss.op_heat_l2_by_op.assign(tick.op_heat_l2_by_op.size(), 0.0F);
        }
        if (loss.op_heat_l2_by_op.size() != tick.op_heat_l2_by_op.size()) {
            throw std::invalid_argument("op heat diagnostics size mismatch");
        }
        for (std::size_t op = 0; op < tick.op_heat_l2_by_op.size(); ++op) {
            loss.op_heat_l2_by_op[op] += tick.op_heat_l2_by_op[op];
        }
    }
    if (tick.chosen_op < loss.op_selection_counts.size()) {
        ++loss.op_selection_counts[tick.chosen_op];
        ++loss.total_selections;
    }
    for (std::size_t rank = 0; rank < tick.candidate_indices.size(); ++rank) {
        if (tick.candidate_indices[rank] == tick.chosen_op) {
            loss.mean_chosen_rank += static_cast<float>(rank);
            break;
        }
    }
    loss.mean_chosen_prob += tick.chosen_prob;
    if (tick.candidate_scores.size() >= 2U) {
        loss.mean_top_score_gap += tick.candidate_scores[0] - tick.candidate_scores[1];
        loss.mean_chosen_score_gap += tick.max_score - tick.chosen_score;
    }
    for (const float probability : tick.candidate_probs) {
        if (probability > 0.0F) {
            loss.mean_candidate_entropy -= probability * std::log(probability);
        }
    }
    if (sample.label >= 0 && sample.class_count > 0 && tick.chosen_op < num_ops) {
        const std::size_t label = static_cast<std::size_t>(sample.label);
        const std::size_t offset = label * num_ops;
        if (offset + tick.chosen_op < loss.class_op_selection_counts.size()) {
            ++loss.class_op_selection_counts[offset + tick.chosen_op];
        }
    }
}

void record_train_result(const TrainResult& result, LossPoint& loss) {
    loss.learning_update_l2 += result.learning_update_l2;
    loss.updated_ops += result.updated_ops;
    if (!result.op_update_l2_by_op.empty()) {
        if (loss.op_train_l2_by_op.empty()) {
            loss.op_train_l2_by_op.assign(result.op_update_l2_by_op.size(), 0.0F);
        }
        if (loss.op_train_l2_by_op.size() != result.op_update_l2_by_op.size()) {
            throw std::invalid_argument("op train diagnostics size mismatch");
        }
        for (std::size_t op = 0; op < result.op_update_l2_by_op.size(); ++op) {
            loss.op_train_l2_by_op[op] += result.op_update_l2_by_op[op];
        }
    }
}

std::vector<float> target_weights_for_frame(const TaskSample& sample, std::size_t state_dim,
                                            std::size_t frame, const TaskConfig& task_config) {
    if (sample.label < 0 || sample.class_count <= 0) {
        return sample.target_weights;
    }

    std::vector<float> weights =
        sample.target_weights.empty() ? uniform_weights(state_dim, 1.0F) : sample.target_weights;
    const std::size_t class_begin = sample.class_offset;
    const std::size_t class_end = class_begin + sample.class_dims;
    if (class_end > weights.size()) {
        throw std::invalid_argument("class target range exceeds target weights");
    }
    if (frame >= task_config.class_start_frame) {
        if (task_config.class_ramp_frames <= 1U) {
            return weights;
        }
        const std::size_t ramp_step = frame - task_config.class_start_frame + 1U;
        const float class_weight_scale =
            std::min(1.0F, static_cast<float>(ramp_step) /
                               static_cast<float>(task_config.class_ramp_frames));
        for (std::size_t dim = class_begin; dim < class_end; ++dim) {
            weights[dim] *= class_weight_scale;
        }
        return weights;
    }
    for (std::size_t dim = class_begin; dim < class_end; ++dim) {
        weights[dim] = 0.0F;
    }
    return weights;
}

void finalize_op_usage(LossPoint& loss) {
    std::size_t selected_ops = 0;
    std::size_t max_op_selections = 0;
    float entropy = 0.0F;
    if (loss.total_selections > 0U) {
        for (const std::size_t count : loss.op_selection_counts) {
            if (count == 0U) {
                continue;
            }
            ++selected_ops;
            max_op_selections = std::max(max_op_selections, count);
            const float p = static_cast<float>(count) / static_cast<float>(loss.total_selections);
            entropy -= p * std::log(p);
        }
        if (selected_ops > 1U) {
            entropy /= std::log(static_cast<float>(loss.op_selection_counts.size()));
        }
    }

    loss.selected_ops = selected_ops;
    loss.max_op_selections = max_op_selections;
    loss.op_selection_entropy = entropy;
    if (loss.total_selections > 0U) {
        const float inv_total = 1.0F / static_cast<float>(loss.total_selections);
        loss.mean_chosen_rank *= inv_total;
        loss.mean_chosen_prob *= inv_total;
        loss.mean_candidate_entropy *= inv_total;
        loss.mean_top_score_gap *= inv_total;
        loss.mean_chosen_score_gap *= inv_total;
    }

    for (std::size_t op = 0; op < loss.op_heat_l2_by_op.size(); ++op) {
        if (loss.op_heat_l2_by_op[op] > loss.max_op_heat_l2) {
            loss.max_op_heat_l2 = loss.op_heat_l2_by_op[op];
            loss.max_op_heat_index = op;
        }
    }
    for (std::size_t op = 0; op < loss.op_train_l2_by_op.size(); ++op) {
        if (loss.op_train_l2_by_op[op] > loss.max_op_train_l2) {
            loss.max_op_train_l2 = loss.op_train_l2_by_op[op];
            loss.max_op_train_index = op;
        }
    }

    const std::size_t num_ops = loss.op_selection_counts.size();
    if (num_ops > 0U && !loss.class_op_selection_counts.empty() &&
        loss.class_op_selection_counts.size() % num_ops == 0U) {
        const std::size_t class_count = loss.class_op_selection_counts.size() / num_ops;
        std::size_t labeled_total = 0;
        std::size_t majority_total = 0;
        for (std::size_t op = 0; op < num_ops; ++op) {
            std::size_t op_total = 0;
            std::size_t op_majority = 0;
            for (std::size_t label = 0; label < class_count; ++label) {
                const std::size_t count = loss.class_op_selection_counts[(label * num_ops) + op];
                op_total += count;
                op_majority = std::max(op_majority, count);
            }
            labeled_total += op_total;
            majority_total += op_majority;
        }
        if (labeled_total > 0U) {
            loss.class_route_purity =
                static_cast<float>(majority_total) / static_cast<float>(labeled_total);
        }
    }
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
    if (task_config.train_interval == 0U) {
        throw std::invalid_argument("train_interval must be nonzero");
    }
    if (task_config.class_start_frame > task_config.frames_per_sample) {
        throw std::invalid_argument("class_start_frame must be <= frames_per_sample");
    }
    if (task_config.learning_rate_decay < 0.0F || task_config.learning_rate_decay > 1.0F) {
        throw std::invalid_argument("learning_rate_decay must be in [0, 1]");
    }
    if (task_config.momentum < 0.0F || task_config.momentum >= 1.0F) {
        throw std::invalid_argument("momentum must be in [0, 1)");
    }
    if (task_config.rejection_decay < 0.0F || task_config.rejection_decay > 1.0F) {
        throw std::invalid_argument("rejection_decay must be in [0, 1]");
    }
    if (task_config.rejection_overuse_scale < 0.0F) {
        throw std::invalid_argument("rejection_overuse_scale must be nonnegative");
    }
    if (task_config.affinity_retain_scale < 0.0F) {
        throw std::invalid_argument("affinity_retain_scale must be nonnegative");
    }
    if (task_config.affinity_retain_threshold < 0.0F) {
        throw std::invalid_argument("affinity_retain_threshold must be nonnegative");
    }
    if (task_config.affinity_retain_underuse_scale < 0.0F) {
        throw std::invalid_argument("affinity_retain_underuse_scale must be nonnegative");
    }
    if (task_config.class_value_scale < 0.0F) {
        throw std::invalid_argument("class_value_scale must be nonnegative");
    }
    if (task_config.class_loss_weight < 0.0F) {
        throw std::invalid_argument("class_loss_weight must be nonnegative");
    }
    if (task_config.world_loss_weight < 0.0F) {
        throw std::invalid_argument("world_loss_weight must be nonnegative");
    }
    if (model_config.state_dim == 0U) {
        throw std::invalid_argument("state_dim must be nonzero");
    }
    if (task_config.task == TaskKind::Mnist) {
        return make_mnist_dataset(model_config, task_config);
    }
    if (task_config.task == TaskKind::Mnist01) {
        return make_mnist_binary_dataset(model_config, task_config);
    }

    std::mt19937 rng(task_config.seed);
    TaskDataset dataset{};
    dataset.train.reserve(task_config.train_samples);
    dataset.test.reserve(task_config.test_samples);

    auto append_samples = [&](std::vector<TaskSample>& samples, std::size_t count,
                              std::size_t offset) {
        for (std::size_t i = 0; i < count; ++i) {
            samples.push_back(
                make_sample(task_config.task, model_config, i, offset, task_config, rng));
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
    case TaskKind::Linear2:
        return "linear-2";
    case TaskKind::Basis4:
        return "basis-4";
    case TaskKind::AlternatingBit:
        return "alternating-bit";
    case TaskKind::Xor:
        return "xor";
    case TaskKind::SineNext:
        return "sine-next";
    case TaskKind::Mnist01:
        return "mnist-01";
    case TaskKind::Mnist:
        return "mnist";
    }
    return "unknown";
}

EvalMetrics evaluate_task_metrics(Model& model, std::span<const TaskSample> samples,
                                  const TaskConfig& task_config) {
    if (samples.empty()) {
        return EvalMetrics{};
    }

    float loss_sum = 0.0F;
    float nonclass_loss_sum = 0.0F;
    float class_loss_sum = 0.0F;
    std::size_t nonclass_loss_samples = 0;
    std::size_t class_loss_samples = 0;
    float class_margin_sum = 0.0F;
    std::size_t correct = 0;
    std::size_t accuracy_samples = 0;
    int max_class_count = 0;
    for (const TaskSample& sample : samples) {
        max_class_count = std::max(max_class_count, sample.class_count);
    }
    std::vector<std::size_t> label_counts(static_cast<std::size_t>(max_class_count), 0U);
    std::vector<std::size_t> prediction_counts(static_cast<std::size_t>(max_class_count), 0U);
    std::vector<std::size_t> correct_counts(static_cast<std::size_t>(max_class_count), 0U);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        validate_sample(samples[i], model.config().state_dim);
        std::vector<float> state = neutral_state(model.config().state_dim);
        for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
            state = model.predict_next(state, samples[i].input);
        }
        if (task_config.task == TaskKind::DelayedCopy) {
            for (std::size_t frame = 0; frame < task_config.idle_frames_between_samples; ++frame) {
                state = model.predict_next(state);
            }
        }

        loss_sum +=
            samples[i].target_weights.empty()
                ? Model::prediction_error(state, samples[i].target)
                : Model::prediction_error(state, samples[i].target, samples[i].target_weights);
        if (samples[i].label >= 0 && samples[i].class_count > 0) {
            float class_error_sum = 0.0F;
            float class_weight_sum = 0.0F;
            float nonclass_error_sum = 0.0F;
            float nonclass_weight_sum = 0.0F;
            const std::size_t class_begin = samples[i].class_offset;
            const std::size_t class_end = class_begin + samples[i].class_dims;
            for (std::size_t dim = 0; dim < state.size(); ++dim) {
                const float weight =
                    samples[i].target_weights.empty() ? 1.0F : samples[i].target_weights[dim];
                const float error = samples[i].target[dim] - state[dim];
                if (dim >= class_begin && dim < class_end) {
                    class_error_sum += weight * error * error;
                    class_weight_sum += weight;
                } else {
                    nonclass_error_sum += weight * error * error;
                    nonclass_weight_sum += weight;
                }
            }
            if (class_weight_sum > 0.0F) {
                class_loss_sum += class_error_sum / class_weight_sum;
                ++class_loss_samples;
            }
            if (nonclass_weight_sum > 0.0F) {
                nonclass_loss_sum += nonclass_error_sum / nonclass_weight_sum;
                ++nonclass_loss_samples;
            }

            const int predicted =
                predicted_class(state, samples[i].class_count, samples[i].class_dims,
                                samples[i].class_offset, task_config.vector_range);
            const float label_score =
                class_score(state, samples[i].class_count, samples[i].class_dims,
                            samples[i].class_offset, task_config.vector_range, samples[i].label);
            float best_other_score = -std::numeric_limits<float>::infinity();
            for (int candidate = 0; candidate < samples[i].class_count; ++candidate) {
                if (candidate == samples[i].label) {
                    continue;
                }
                best_other_score = std::max(
                    best_other_score,
                    class_score(state, samples[i].class_count, samples[i].class_dims,
                                samples[i].class_offset, task_config.vector_range, candidate));
            }
            class_margin_sum += label_score - best_other_score;
            const bool is_correct = predicted == samples[i].label;
            correct += is_correct ? 1U : 0U;
            if (static_cast<std::size_t>(samples[i].label) < label_counts.size()) {
                ++label_counts[static_cast<std::size_t>(samples[i].label)];
                if (is_correct) {
                    ++correct_counts[static_cast<std::size_t>(samples[i].label)];
                }
            }
            if (predicted >= 0 && static_cast<std::size_t>(predicted) < prediction_counts.size()) {
                ++prediction_counts[static_cast<std::size_t>(predicted)];
            }
            ++accuracy_samples;
        }
    }
    float balanced_accuracy = 0.0F;
    std::size_t balanced_classes = 0;
    for (std::size_t label = 0; label < label_counts.size(); ++label) {
        if (label_counts[label] == 0U) {
            continue;
        }
        balanced_accuracy +=
            static_cast<float>(correct_counts[label]) / static_cast<float>(label_counts[label]);
        ++balanced_classes;
    }
    if (balanced_classes > 0U) {
        balanced_accuracy /= static_cast<float>(balanced_classes);
    }

    return EvalMetrics{
        .loss = loss_sum / static_cast<float>(samples.size()),
        .nonclass_loss = nonclass_loss_samples == 0U
                             ? 0.0F
                             : nonclass_loss_sum / static_cast<float>(nonclass_loss_samples),
        .class_loss = class_loss_samples == 0U
                          ? 0.0F
                          : class_loss_sum / static_cast<float>(class_loss_samples),
        .accuracy = accuracy_samples == 0U
                        ? 0.0F
                        : static_cast<float>(correct) / static_cast<float>(accuracy_samples),
        .balanced_accuracy = balanced_accuracy,
        .mean_class_margin =
            accuracy_samples == 0U ? 0.0F : class_margin_sum / static_cast<float>(accuracy_samples),
        .accuracy_samples = accuracy_samples,
        .label_counts = std::move(label_counts),
        .prediction_counts = std::move(prediction_counts),
        .correct_counts = std::move(correct_counts),
    };
}

float evaluate_task_loss(Model& model, std::span<const TaskSample> samples,
                         const TaskConfig& task_config) {
    return evaluate_task_metrics(model, samples, task_config).loss;
}

LossPoint train_task_epoch(Model& model, std::span<const TaskSample> train_samples,
                           std::span<const TaskSample> test_samples, const TaskConfig& task_config,
                           std::size_t epoch, std::span<const float> op_anchor) {
    if (train_samples.empty()) {
        const EvalMetrics metrics = evaluate_task_metrics(model, test_samples, task_config);
        LossPoint loss{};
        loss.test_loss = metrics.loss;
        loss.test_nonclass_loss = metrics.nonclass_loss;
        loss.test_class_loss = metrics.class_loss;
        loss.test_accuracy = metrics.accuracy;
        loss.test_balanced_accuracy = metrics.balanced_accuracy;
        loss.accuracy_samples = metrics.accuracy_samples;
        loss.label_counts = metrics.label_counts;
        loss.prediction_counts = metrics.prediction_counts;
        loss.correct_counts = metrics.correct_counts;
        loss.op_selection_counts.assign(model.config().num_ops, 0U);
        return loss;
    }

    TrainConfig train_config{};
    train_config.learning_rate =
        task_config.learning_rate *
        std::pow(task_config.learning_rate_decay, static_cast<float>(epoch));
    train_config.momentum = task_config.momentum;
    train_config.recency_decay = task_config.recency_decay;
    train_config.max_grad_norm = task_config.max_grad_norm;
    train_config.rejection_scale = task_config.rejection_scale *
                                   std::pow(task_config.rejection_decay, static_cast<float>(epoch));
    train_config.rejection_threshold = task_config.rejection_threshold;
    train_config.rejection_overuse_scale = task_config.rejection_overuse_scale;
    train_config.affinity_retain_scale = task_config.affinity_retain_scale;
    train_config.affinity_retain_threshold = task_config.affinity_retain_threshold;
    train_config.affinity_retain_underuse_scale = task_config.affinity_retain_underuse_scale;
    train_config.op_anchor = op_anchor;
    train_config.op_anchor_scale = task_config.op_anchor_scale;
    train_config.backprop_through_state = task_config.backprop_through_state;

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
    LossPoint diagnostics{};
    diagnostics.op_selection_counts.assign(model.config().num_ops, 0U);
    diagnostics.op_heat_l2_by_op.assign(model.config().num_ops, 0.0F);
    diagnostics.op_train_l2_by_op.assign(model.config().num_ops, 0.0F);
    int max_class_count = 0;
    for (const TaskSample& sample : train_samples) {
        max_class_count = std::max(max_class_count, sample.class_count);
    }
    if (max_class_count > 0) {
        diagnostics.class_op_selection_counts.assign(
            static_cast<std::size_t>(max_class_count) * model.config().num_ops, 0U);
    }
    train_config.op_usage_counts = diagnostics.op_selection_counts;

    for (std::size_t order_index = 0; order_index < order.size(); ++order_index) {
        const TaskSample& sample = train_samples[order[order_index]];
        validate_sample(sample, model.config().state_dim);

        std::vector<float> state = neutral_state(model.config().state_dim);
        std::vector<Tick> window;
        window.reserve(task_config.window_size);
        std::size_t sample_ticks = 0;
        auto train_current_window = [&](bool force) {
            if (window.empty()) {
                return;
            }
            if (!force && sample_ticks % task_config.train_interval != 0U) {
                return;
            }
            record_train_result(model.train_window(window, train_config), diagnostics);
        };

        for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
            const std::size_t clock =
                ((epoch * train_samples.size()) + order_index) * task_config.frames_per_sample +
                frame;
            Tick tick = model.tick(state, rng, clock, sample.input);
            const std::vector<float> frame_target_weights =
                target_weights_for_frame(sample, model.config().state_dim, frame, task_config);
            apply_observation(tick, sample.target, model.config().curiosity_scale,
                              frame_target_weights);
            record_tick_diagnostics(tick, sample, model.config().num_ops, diagnostics);

            train_loss_sum += tick.prediction_error;
            ++trained_ticks;

            if (window.size() == task_config.window_size) {
                window.erase(window.begin());
            }
            window.push_back(std::move(tick));
            ++sample_ticks;
            train_current_window(false);
        }

        for (std::size_t frame = 0; frame < task_config.idle_frames_between_samples; ++frame) {
            const std::size_t clock =
                (((epoch * train_samples.size()) + order_index) *
                 (task_config.frames_per_sample + task_config.idle_frames_between_samples)) +
                task_config.frames_per_sample + frame;
            Tick tick = model.tick(state, rng, clock);
            if (task_config.task == TaskKind::DelayedCopy) {
                apply_observation(tick, sample.target, model.config().curiosity_scale,
                                  sample.target_weights);
            }
            record_tick_diagnostics(tick, sample, model.config().num_ops, diagnostics);

            self_loss_sum += tick.prediction_error;
            ++self_ticks;

            if (window.size() == task_config.window_size) {
                window.erase(window.begin());
            }
            window.push_back(std::move(tick));
            ++sample_ticks;
            train_current_window(false);
        }
        if (sample_ticks % task_config.train_interval != 0U) {
            train_current_window(true);
        }
    }

    const EvalMetrics metrics = evaluate_task_metrics(model, test_samples, task_config);
    diagnostics.train_loss = train_loss_sum / static_cast<float>(trained_ticks);
    diagnostics.self_loss =
        self_ticks == 0U ? 0.0F : self_loss_sum / static_cast<float>(self_ticks);
    diagnostics.test_loss = metrics.loss;
    diagnostics.test_nonclass_loss = metrics.nonclass_loss;
    diagnostics.test_class_loss = metrics.class_loss;
    diagnostics.test_accuracy = metrics.accuracy;
    diagnostics.test_balanced_accuracy = metrics.balanced_accuracy;
    diagnostics.mean_class_margin = metrics.mean_class_margin;
    diagnostics.accuracy_samples = metrics.accuracy_samples;
    diagnostics.label_counts = metrics.label_counts;
    diagnostics.prediction_counts = metrics.prediction_counts;
    diagnostics.correct_counts = metrics.correct_counts;
    finalize_op_usage(diagnostics);
    return diagnostics;
}

} // namespace vvm
