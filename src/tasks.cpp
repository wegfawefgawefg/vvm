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

TaskSample make_mnist_sample(std::span<const unsigned char, 784> pixels, unsigned char label,
                             std::size_t state_dim, VectorRange range) {
    constexpr std::size_t kImageDims = 784;
    constexpr std::size_t kDigitOffset = 784;
    constexpr std::size_t kDigitClasses = 10;
    if (state_dim < kImageDims + kDigitClasses) {
        throw std::invalid_argument("mnist task requires state_dim >= 794");
    }

    std::vector<float> input(state_dim, 0.0F);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        input[i] = static_cast<float>(pixels[i]) / 255.0F;
    }
    normalize_l2(input);

    std::vector<float> target(state_dim, 0.0F);
    std::vector<float> target_weights = uniform_weights(state_dim, 1.0F);
    constexpr float kImageWeight = 1.0F;
    constexpr float kClassWeight = 2.0F;
    constexpr float kClassTargetWeight = 64.0F;
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        target[i] = kImageWeight * input[i];
    }
    const std::vector<float> digit = class_vector(kDigitClasses, static_cast<int>(label),
                                                  static_cast<int>(kDigitClasses), range);
    for (std::size_t i = 0; i < digit.size(); ++i) {
        target[kDigitOffset + i] = kClassWeight * digit[i];
        target_weights[kDigitOffset + i] = kClassTargetWeight;
    }
    normalize_l2(target);

    return TaskSample{
        .input = std::move(input),
        .target = std::move(target),
        .target_weights = std::move(target_weights),
        .label = static_cast<int>(label),
        .class_count = static_cast<int>(kDigitClasses),
        .class_offset = kDigitOffset,
    };
}

TaskSample make_mnist_binary_sample(std::span<const unsigned char, 784> pixels, unsigned char label,
                                    std::size_t state_dim, VectorRange range) {
    constexpr std::size_t kImageDims = 784;
    constexpr std::size_t kDigitOffset = 784;
    constexpr std::size_t kDigitClasses = 2;
    if (state_dim < kImageDims + kDigitClasses) {
        throw std::invalid_argument("mnist-01 task requires state_dim >= 786");
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
    std::vector<float> target_weights = uniform_weights(state_dim, 1.0F);
    constexpr float kImageWeight = 1.0F;
    constexpr float kClassWeight = 2.0F;
    constexpr float kClassTargetWeight = 128.0F;
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        target[i] = kImageWeight * input[i];
    }
    const std::vector<float> digit = class_vector(kDigitClasses, static_cast<int>(label),
                                                  static_cast<int>(kDigitClasses), range);
    for (std::size_t i = 0; i < digit.size(); ++i) {
        target[kDigitOffset + i] = kClassWeight * digit[i];
        target_weights[kDigitOffset + i] = kClassTargetWeight;
    }
    normalize_l2(target);

    return TaskSample{
        .input = std::move(input),
        .target = std::move(target),
        .target_weights = std::move(target_weights),
        .label = static_cast<int>(label),
        .class_count = static_cast<int>(kDigitClasses),
        .class_offset = kDigitOffset,
    };
}

void append_mnist_split(std::vector<TaskSample>& samples, const std::filesystem::path& images_path,
                        const std::filesystem::path& labels_path, std::size_t requested_count,
                        std::size_t state_dim, VectorRange range) {
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
        samples.push_back(make_mnist_sample(pixels, label, state_dim, range));
    }
}

void append_mnist_binary_split(std::vector<TaskSample>& samples,
                               const std::filesystem::path& images_path,
                               const std::filesystem::path& labels_path,
                               std::size_t requested_count, std::size_t state_dim,
                               VectorRange range) {
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

    std::array<unsigned char, 784> pixels = {};
    for (std::size_t i = 0; i < image_count && samples.size() < requested_count; ++i) {
        unsigned char label = 0;
        images.read(reinterpret_cast<char*>(pixels.data()),
                    static_cast<std::streamsize>(pixels.size()));
        labels.read(reinterpret_cast<char*>(&label), 1);
        if (!images || !labels) {
            throw std::runtime_error("truncated MNIST IDX files in " +
                                     images_path.parent_path().string());
        }
        if (label <= 1U) {
            samples.push_back(make_mnist_binary_sample(pixels, label, state_dim, range));
        }
    }
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
        constexpr float kClassWeight = 2.0F;
        constexpr float kClassTargetWeight = 16.0F;
        for (std::size_t i = 0; i < class_target.size(); ++i) {
            target[class_offset + i] = kClassWeight * class_target[i];
            target_weights[class_offset + i] = kClassTargetWeight;
        }
        normalize_l2(target);
        return TaskSample{
            .input = std::move(input),
            .target = std::move(target),
            .target_weights = std::move(target_weights),
            .label = label ? 1 : 0,
            .class_count = 2,
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
                       model_config.state_dim, task_config.vector_range);
    append_mnist_split(dataset.test, root / "t10k-images-idx3-ubyte",
                       root / "t10k-labels-idx1-ubyte", task_config.test_samples,
                       model_config.state_dim, task_config.vector_range);
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
                              model_config.state_dim, task_config.vector_range);
    append_mnist_binary_split(dataset.test, root / "t10k-images-idx3-ubyte",
                              root / "t10k-labels-idx1-ubyte", task_config.test_samples,
                              model_config.state_dim, task_config.vector_range);
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
}

int predicted_class(std::span<const float> state, int class_count, std::size_t class_offset,
                    VectorRange range) {
    if (class_count <= 0 || class_offset + static_cast<std::size_t>(class_count) > state.size()) {
        throw std::invalid_argument("invalid class count");
    }

    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int candidate = 0; candidate < class_count; ++candidate) {
        float score = 0.0F;
        for (int i = 0; i < class_count; ++i) {
            const bool matches = i == candidate;
            const float value = state[class_offset + static_cast<std::size_t>(i)];
            if (range == VectorRange::Signed) {
                const float off_value =
                    class_count <= 2 ? -1.0F : -2.0F / static_cast<float>(class_count - 2);
                score += value * (matches ? 1.0F : off_value);
            } else if (matches) {
                score += value;
            }
        }
        if (score > best_score) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

void record_tick_diagnostics(const Tick& tick, LossPoint& loss) {
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

    std::mt19937 rng(task_config.seed ^ 0xBADC0DEU);
    float loss_sum = 0.0F;
    std::size_t correct = 0;
    std::size_t accuracy_samples = 0;
    int max_class_count = 0;
    for (const TaskSample& sample : samples) {
        max_class_count = std::max(max_class_count, sample.class_count);
    }
    std::vector<std::size_t> label_counts(static_cast<std::size_t>(max_class_count), 0U);
    std::vector<std::size_t> prediction_counts(static_cast<std::size_t>(max_class_count), 0U);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        validate_sample(samples[i], model.config().state_dim);
        std::vector<float> state = neutral_state(model.config().state_dim);
        for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
            (void)model.tick(state, rng, (i * task_config.frames_per_sample) + frame,
                             samples[i].input);
        }
        if (task_config.task == TaskKind::DelayedCopy) {
            for (std::size_t frame = 0; frame < task_config.idle_frames_between_samples; ++frame) {
                (void)model.tick(state, rng,
                                 (i * task_config.frames_per_sample) +
                                     task_config.frames_per_sample + frame);
            }
        }

        loss_sum +=
            samples[i].target_weights.empty()
                ? Model::prediction_error(state, samples[i].target)
                : Model::prediction_error(state, samples[i].target, samples[i].target_weights);
        if (samples[i].label >= 0 && samples[i].class_count > 0) {
            const int predicted = predicted_class(
                state, samples[i].class_count, samples[i].class_offset, task_config.vector_range);
            correct += predicted == samples[i].label ? 1U : 0U;
            if (static_cast<std::size_t>(samples[i].label) < label_counts.size()) {
                ++label_counts[static_cast<std::size_t>(samples[i].label)];
            }
            if (predicted >= 0 && static_cast<std::size_t>(predicted) < prediction_counts.size()) {
                ++prediction_counts[static_cast<std::size_t>(predicted)];
            }
            ++accuracy_samples;
        }
    }
    return EvalMetrics{
        .loss = loss_sum / static_cast<float>(samples.size()),
        .accuracy = accuracy_samples == 0U
                        ? 0.0F
                        : static_cast<float>(correct) / static_cast<float>(accuracy_samples),
        .accuracy_samples = accuracy_samples,
        .label_counts = std::move(label_counts),
        .prediction_counts = std::move(prediction_counts),
    };
}

float evaluate_task_loss(Model& model, std::span<const TaskSample> samples,
                         const TaskConfig& task_config) {
    return evaluate_task_metrics(model, samples, task_config).loss;
}

LossPoint train_task_epoch(Model& model, std::span<const TaskSample> train_samples,
                           std::span<const TaskSample> test_samples, const TaskConfig& task_config,
                           std::size_t epoch) {
    if (train_samples.empty()) {
        const EvalMetrics metrics = evaluate_task_metrics(model, test_samples, task_config);
        LossPoint loss{};
        loss.test_loss = metrics.loss;
        loss.test_accuracy = metrics.accuracy;
        loss.accuracy_samples = metrics.accuracy_samples;
        loss.op_selection_counts.assign(model.config().num_ops, 0U);
        return loss;
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
    LossPoint diagnostics{};
    diagnostics.op_selection_counts.assign(model.config().num_ops, 0U);
    diagnostics.op_heat_l2_by_op.assign(model.config().num_ops, 0.0F);
    diagnostics.op_train_l2_by_op.assign(model.config().num_ops, 0.0F);

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
            apply_observation(tick, sample.target, model.config().curiosity_scale,
                              sample.target_weights);
            record_tick_diagnostics(tick, diagnostics);

            train_loss_sum += tick.prediction_error;
            ++trained_ticks;

            if (window.size() == task_config.window_size) {
                window.erase(window.begin());
            }
            window.push_back(std::move(tick));

            record_train_result(model.train_window(window, train_config), diagnostics);
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
            record_tick_diagnostics(tick, diagnostics);

            self_loss_sum += tick.prediction_error;
            ++self_ticks;

            if (window.size() == task_config.window_size) {
                window.erase(window.begin());
            }
            window.push_back(std::move(tick));

            record_train_result(model.train_window(window, train_config), diagnostics);
        }
    }

    const EvalMetrics metrics = evaluate_task_metrics(model, test_samples, task_config);
    diagnostics.train_loss = train_loss_sum / static_cast<float>(trained_ticks);
    diagnostics.self_loss =
        self_ticks == 0U ? 0.0F : self_loss_sum / static_cast<float>(self_ticks);
    diagnostics.test_loss = metrics.loss;
    diagnostics.test_accuracy = metrics.accuracy;
    diagnostics.accuracy_samples = metrics.accuracy_samples;
    diagnostics.label_counts = metrics.label_counts;
    diagnostics.prediction_counts = metrics.prediction_counts;
    finalize_op_usage(diagnostics);
    return diagnostics;
}

} // namespace vvm
