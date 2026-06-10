#include "vvm/model.hpp"
#include "vvm/readout.hpp"
#include "vvm/tasks.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef VVM_WITH_SDL3
namespace vvm {
int run_visualizer(const Config& config);
int run_training_visualizer(const Config& config, const TaskConfig& task_config,
                            std::size_t epochs);
} // namespace vvm
#endif

namespace {

enum class ReadoutSource {
    Input,
    Vvm,
};

struct CliOptions {
    std::size_t epochs = 200;
    std::size_t hidden_dim = 64;
    float readout_learning_rate = 0.1F;
    std::string output_dir = "artifacts/reconstructions";
    ReadoutSource readout_source = ReadoutSource::Input;
    bool restore_best = false;
    bool anchor_to_best = false;
};

void print_usage() {
    std::cout << "usage:\n"
              << "  vvm smoke\n"
              << "  vvm run [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--activation deadzone] [--update-scale F] "
                 "[--retrieval-temperature F] [--state-heat F] [--op-heat F] "
                 "[--heat-decay F] [--sample-candidates N] [--hard-retrieval]\n"
              << "  vvm train-task [--epochs N] [--train-samples N] [--test-samples N] "
                 "[--task copy-input|delayed-copy|linear-2|basis-4|alternating-bit|xor|"
                 "sine-next|mnist-01|mnist] "
                 "[--mnist-dir PATH] [--vectors signed|nonnegative] [--sample-frames N] "
                 "[--idle-frames N] [--window N] [--train-interval N] "
                 "[--sample-candidates N] "
                 "[--class-start-frame N] [--class-ramp-frames N] "
                 "[--class-registers N] [--lr F] [--lr-decay F] "
                 "[--momentum F] [--class-loss-weight F] "
                 "[--world-loss-weight F] [--class-value-scale F] [--rejection-decay F] "
                 "[--rejection-overuse-scale F] "
                 "[--affinity-retain-scale F] "
                 "[--affinity-retain-threshold F] [--affinity-retain-underuse-scale F] "
                 "[--op-anchor-scale F] [--anchor-to-best] [--bptt] [--restore-best]\n"
              << "  vvm train-readout [--task mnist] [--readout-source input|vvm] "
                 "[--readout-lr F] [--epochs N] [--train-samples N] [--test-samples N]\n"
              << "  vvm train-mlp [--task mnist-01|mnist] [--hidden N] [--readout-lr F] "
                 "[--epochs N] [--train-samples N] [--test-samples N]\n"
              << "  vvm train-autoencoder [--task mnist-01|mnist] [--hidden N] "
                 "[--readout-lr F] [--epochs N] [--train-samples N] [--test-samples N]\n"
              << "  vvm export-reconstructions [--output-dir PATH] [--hidden N] "
                 "[--readout-lr F] [train-task options]\n"
              << "  vvm bench-tasks [--epochs N] [--state-dim N] [--ops N] [--candidates N]\n"
              << "  vvm visualize [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--update-scale F] [--state-heat F] [--op-heat F] [--heat-decay F]\n"
              << "  vvm visualize-train [--epochs N] [--train-samples N] [--test-samples N] "
                 "[--sample-frames N] [--idle-frames N] [--window N] [--lr F]\n";
    std::cout << "notes:\n"
              << "  deadzone is the main VVM activation. relu, leaky-relu, and clamp are "
                 "ablation/control modes.\n";
}

bool parse_size(std::string_view value, std::size_t& out) {
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    const auto parsed = std::from_chars(begin, end, out);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_float(std::string_view value, float& out) {
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    const auto parsed = std::from_chars(begin, end, out);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_task(std::string_view value, vvm::TaskKind& out) {
    if (value == "copy" || value == "copy-input") {
        out = vvm::TaskKind::CopyInput;
        return true;
    }
    if (value == "delayed-copy" || value == "delay" || value == "delayed") {
        out = vvm::TaskKind::DelayedCopy;
        return true;
    }
    if (value == "linear-2" || value == "linear2" || value == "linearly-separable") {
        out = vvm::TaskKind::Linear2;
        return true;
    }
    if (value == "basis-4" || value == "basis4" || value == "four-class-basis") {
        out = vvm::TaskKind::Basis4;
        return true;
    }
    if (value == "alternating-bit" || value == "alternating" || value == "alt-bit") {
        out = vvm::TaskKind::AlternatingBit;
        return true;
    }
    if (value == "xor") {
        out = vvm::TaskKind::Xor;
        return true;
    }
    if (value == "sine-next" || value == "sine") {
        out = vvm::TaskKind::SineNext;
        return true;
    }
    if (value == "mnist-01" || value == "mnist01" || value == "mnist-0-1") {
        out = vvm::TaskKind::Mnist01;
        return true;
    }
    if (value == "mnist") {
        out = vvm::TaskKind::Mnist;
        return true;
    }
    return false;
}

bool parse_activation(std::string_view value, vvm::ActivationKind& out) {
    if (value == "relu") {
        out = vvm::ActivationKind::Relu;
        return true;
    }
    if (value == "leaky-relu" || value == "leaky") {
        out = vvm::ActivationKind::LeakyRelu;
        return true;
    }
    if (value == "clamp") {
        out = vvm::ActivationKind::Clamp;
        return true;
    }
    if (value == "deadzone" || value == "hardshrink" || value == "signed-threshold") {
        out = vvm::ActivationKind::Deadzone;
        return true;
    }
    return false;
}

bool parse_vector_range(std::string_view value, vvm::VectorRange& out) {
    if (value == "signed") {
        out = vvm::VectorRange::Signed;
        return true;
    }
    if (value == "nonnegative" || value == "positive") {
        out = vvm::VectorRange::Nonnegative;
        return true;
    }
    return false;
}

bool parse_readout_source(std::string_view value, ReadoutSource& out) {
    if (value == "input" || value == "raw") {
        out = ReadoutSource::Input;
        return true;
    }
    if (value == "vvm" || value == "state") {
        out = ReadoutSource::Vvm;
        return true;
    }
    return false;
}

const char* activation_name(vvm::ActivationKind activation) {
    switch (activation) {
    case vvm::ActivationKind::Relu:
        return "relu";
    case vvm::ActivationKind::LeakyRelu:
        return "leaky-relu";
    case vvm::ActivationKind::Clamp:
        return "clamp";
    case vvm::ActivationKind::Deadzone:
        return "deadzone";
    }
    return "unknown";
}

const char* vector_range_name(vvm::VectorRange range) {
    switch (range) {
    case vvm::VectorRange::Signed:
        return "signed";
    case vvm::VectorRange::Nonnegative:
        return "nonnegative";
    }
    return "unknown";
}

bool parse_options(std::span<char*> args, vvm::Config& config, vvm::TaskConfig& task_config,
                   CliOptions& cli_options) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg(args[i]);
        if (arg == "--hard-retrieval" || arg == "--greedy-retrieval") {
            config.sample_retrieval = false;
            continue;
        }
        if (arg == "--bptt" || arg == "--backprop-through-state") {
            task_config.backprop_through_state = true;
            continue;
        }
        if (arg == "--local-bp" || arg == "--local-backprop") {
            task_config.backprop_through_state = false;
            continue;
        }
        if (arg == "--sample-retrieval") {
            config.sample_retrieval = true;
            continue;
        }
        if (arg == "--restore-best") {
            cli_options.restore_best = true;
            continue;
        }
        if (arg == "--anchor-to-best") {
            cli_options.anchor_to_best = true;
            continue;
        }
        if (i + 1 >= args.size()) {
            std::cerr << "missing value for " << arg << '\n';
            return false;
        }

        const std::string_view value(args[i + 1]);
        if (arg == "--task") {
            if (!parse_task(value, task_config.task)) {
                return false;
            }
        } else if (arg == "--activation") {
            if (!parse_activation(value, config.activation)) {
                return false;
            }
        } else if (arg == "--vectors" || arg == "--vector-range") {
            if (!parse_vector_range(value, task_config.vector_range)) {
                return false;
            }
        } else if (arg == "--mnist-dir") {
            task_config.mnist_dir = std::string(value);
        } else if (arg == "--steps") {
            if (!parse_size(value, config.steps)) {
                return false;
            }
        } else if (arg == "--state-dim") {
            if (!parse_size(value, config.state_dim)) {
                return false;
            }
        } else if (arg == "--ops") {
            if (!parse_size(value, config.num_ops)) {
                return false;
            }
        } else if (arg == "--candidates" || arg == "--top-k") {
            if (!parse_size(value, config.candidate_count)) {
                return false;
            }
        } else if (arg == "--sample-candidates") {
            if (!parse_size(value, config.sample_candidate_count)) {
                return false;
            }
        } else if (arg == "--update-scale") {
            if (!parse_float(value, config.update_scale)) {
                return false;
            }
        } else if (arg == "--input-scale") {
            if (!parse_float(value, config.input_scale)) {
                return false;
            }
        } else if (arg == "--activation-threshold") {
            if (!parse_float(value, config.activation_threshold)) {
                return false;
            }
        } else if (arg == "--activation-leak") {
            if (!parse_float(value, config.activation_leak)) {
                return false;
            }
        } else if (arg == "--retrieval-temperature") {
            if (!parse_float(value, config.retrieval_temperature)) {
                return false;
            }
        } else if (arg == "--state-heat") {
            if (!parse_float(value, config.state_heat_stddev)) {
                return false;
            }
        } else if (arg == "--op-heat") {
            if (!parse_float(value, config.op_heat_stddev)) {
                return false;
            }
        } else if (arg == "--heat-decay") {
            if (!parse_float(value, config.heat_decay)) {
                return false;
            }
        } else if (arg == "--curiosity-scale") {
            if (!parse_float(value, config.curiosity_scale)) {
                return false;
            }
        } else if (arg == "--epochs") {
            if (!parse_size(value, cli_options.epochs)) {
                return false;
            }
        } else if (arg == "--hidden" || arg == "--hidden-dim") {
            if (!parse_size(value, cli_options.hidden_dim)) {
                return false;
            }
        } else if (arg == "--readout-lr") {
            if (!parse_float(value, cli_options.readout_learning_rate)) {
                return false;
            }
        } else if (arg == "--output-dir") {
            cli_options.output_dir = std::string(value);
        } else if (arg == "--readout-source") {
            if (!parse_readout_source(value, cli_options.readout_source)) {
                return false;
            }
        } else if (arg == "--train-samples") {
            if (!parse_size(value, task_config.train_samples)) {
                return false;
            }
        } else if (arg == "--test-samples") {
            if (!parse_size(value, task_config.test_samples)) {
                return false;
            }
        } else if (arg == "--sample-frames" || arg == "--frames-per-sample") {
            if (!parse_size(value, task_config.frames_per_sample)) {
                return false;
            }
        } else if (arg == "--idle-frames") {
            if (!parse_size(value, task_config.idle_frames_between_samples)) {
                return false;
            }
        } else if (arg == "--window") {
            if (!parse_size(value, task_config.window_size)) {
                return false;
            }
        } else if (arg == "--train-interval") {
            if (!parse_size(value, task_config.train_interval)) {
                return false;
            }
        } else if (arg == "--class-start-frame") {
            if (!parse_size(value, task_config.class_start_frame)) {
                return false;
            }
        } else if (arg == "--class-ramp-frames") {
            if (!parse_size(value, task_config.class_ramp_frames)) {
                return false;
            }
        } else if (arg == "--class-registers") {
            if (!parse_size(value, task_config.class_registers)) {
                return false;
            }
        } else if (arg == "--lr" || arg == "--learning-rate") {
            if (!parse_float(value, task_config.learning_rate)) {
                return false;
            }
        } else if (arg == "--lr-decay" || arg == "--learning-rate-decay") {
            if (!parse_float(value, task_config.learning_rate_decay)) {
                return false;
            }
        } else if (arg == "--momentum") {
            if (!parse_float(value, task_config.momentum)) {
                return false;
            }
        } else if (arg == "--recency-decay") {
            if (!parse_float(value, task_config.recency_decay)) {
                return false;
            }
        } else if (arg == "--max-grad-norm") {
            if (!parse_float(value, task_config.max_grad_norm)) {
                return false;
            }
        } else if (arg == "--rejection-scale") {
            if (!parse_float(value, task_config.rejection_scale)) {
                return false;
            }
        } else if (arg == "--rejection-threshold") {
            if (!parse_float(value, task_config.rejection_threshold)) {
                return false;
            }
        } else if (arg == "--rejection-decay") {
            if (!parse_float(value, task_config.rejection_decay)) {
                return false;
            }
        } else if (arg == "--rejection-overuse-scale") {
            if (!parse_float(value, task_config.rejection_overuse_scale)) {
                return false;
            }
        } else if (arg == "--affinity-retain-scale") {
            if (!parse_float(value, task_config.affinity_retain_scale)) {
                return false;
            }
        } else if (arg == "--affinity-retain-threshold") {
            if (!parse_float(value, task_config.affinity_retain_threshold)) {
                return false;
            }
        } else if (arg == "--affinity-retain-underuse-scale") {
            if (!parse_float(value, task_config.affinity_retain_underuse_scale)) {
                return false;
            }
        } else if (arg == "--op-anchor-scale") {
            if (!parse_float(value, task_config.op_anchor_scale)) {
                return false;
            }
        } else if (arg == "--class-value-scale") {
            if (!parse_float(value, task_config.class_value_scale)) {
                return false;
            }
        } else if (arg == "--world-loss-weight") {
            if (!parse_float(value, task_config.world_loss_weight)) {
                return false;
            }
        } else if (arg == "--class-loss-weight") {
            if (!parse_float(value, task_config.class_loss_weight)) {
                return false;
            }
        } else {
            std::cerr << "unknown option: " << arg << '\n';
            return false;
        }
        ++i;
    }
    return true;
}

const char* readout_source_name(ReadoutSource source) {
    switch (source) {
    case ReadoutSource::Input:
        return "input";
    case ReadoutSource::Vvm:
        return "vvm";
    }
    return "unknown";
}

void print_counts(std::string_view name, std::span<const std::size_t> counts) {
    if (counts.empty()) {
        return;
    }
    std::cout << ' ' << name << "=[";
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (i > 0U) {
            std::cout << ',';
        }
        std::cout << counts[i];
    }
    std::cout << ']';
}

void print_recalls(std::span<const std::size_t> correct_counts,
                   std::span<const std::size_t> label_counts) {
    if (correct_counts.empty() || correct_counts.size() != label_counts.size()) {
        return;
    }
    std::cout << " recall=[";
    for (std::size_t i = 0; i < correct_counts.size(); ++i) {
        if (i > 0U) {
            std::cout << ',';
        }
        const float recall = label_counts[i] == 0U ? 0.0F
                                                   : static_cast<float>(correct_counts[i]) /
                                                         static_cast<float>(label_counts[i]);
        std::cout << (100.0F * recall) << "%";
    }
    std::cout << ']';
}

void print_top_counts(std::string_view name, std::span<const std::size_t> counts,
                      std::size_t limit = 3U) {
    if (counts.empty() || limit == 0U) {
        return;
    }

    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    ranked.reserve(counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] > 0U) {
            ranked.emplace_back(counts[i], i);
        }
    }
    if (ranked.empty()) {
        return;
    }

    const std::size_t shown = std::min(limit, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(shown),
                      ranked.end(), [](const auto& lhs, const auto& rhs) {
                          if (lhs.first != rhs.first) {
                              return lhs.first > rhs.first;
                          }
                          return lhs.second < rhs.second;
                      });

    std::cout << ' ' << name << "=[";
    for (std::size_t i = 0; i < shown; ++i) {
        if (i > 0U) {
            std::cout << ',';
        }
        std::cout << ranked[i].second << ':' << ranked[i].first;
    }
    std::cout << ']';
}

void print_top_floats(std::string_view name, std::span<const float> values,
                      std::size_t limit = 3U) {
    if (values.empty() || limit == 0U) {
        return;
    }

    std::vector<std::pair<float, std::size_t>> ranked;
    ranked.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] > 0.0F) {
            ranked.emplace_back(values[i], i);
        }
    }
    if (ranked.empty()) {
        return;
    }

    const std::size_t shown = std::min(limit, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(shown),
                      ranked.end(), [](const auto& lhs, const auto& rhs) {
                          if (lhs.first != rhs.first) {
                              return lhs.first > rhs.first;
                          }
                          return lhs.second < rhs.second;
                      });

    std::cout << ' ' << name << "=[";
    for (std::size_t i = 0; i < shown; ++i) {
        if (i > 0U) {
            std::cout << ',';
        }
        std::cout << ranked[i].second << ':' << ranked[i].first;
    }
    std::cout << ']';
}

void print_class_top_counts(std::span<const std::size_t> counts, std::size_t class_count,
                            std::size_t num_ops, std::size_t limit = 3U,
                            std::size_t max_classes = 4U) {
    if (counts.empty() || class_count == 0U || num_ops == 0U ||
        counts.size() < class_count * num_ops) {
        return;
    }

    const std::size_t shown_classes = std::min(class_count, max_classes);
    std::cout << " class_top=[";
    for (std::size_t label = 0; label < shown_classes; ++label) {
        if (label > 0U) {
            std::cout << ';';
        }
        std::cout << label << ':';
        const std::span<const std::size_t> class_counts(counts.data() + (label * num_ops), num_ops);
        std::vector<std::pair<std::size_t, std::size_t>> ranked;
        ranked.reserve(num_ops);
        for (std::size_t op = 0; op < num_ops; ++op) {
            if (class_counts[op] > 0U) {
                ranked.emplace_back(class_counts[op], op);
            }
        }
        const std::size_t shown = std::min(limit, ranked.size());
        if (shown == 0U) {
            std::cout << '-';
            continue;
        }
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(shown),
                          ranked.end(), [](const auto& lhs, const auto& rhs) {
                              if (lhs.first != rhs.first) {
                                  return lhs.first > rhs.first;
                              }
                              return lhs.second < rhs.second;
                          });
        for (std::size_t i = 0; i < shown; ++i) {
            if (i > 0U) {
                std::cout << ',';
            }
            std::cout << ranked[i].second << ':' << ranked[i].first;
        }
    }
    std::cout << ']';
}

std::size_t infer_class_count(std::span<const vvm::TaskSample> samples) {
    int class_count = 0;
    for (const vvm::TaskSample& sample : samples) {
        if (sample.class_count > class_count) {
            class_count = sample.class_count;
        }
        if (sample.label >= 0) {
            class_count = std::max(class_count, sample.label + 1);
        }
    }
    if (class_count <= 0) {
        throw std::invalid_argument("readout training requires labeled samples");
    }
    return static_cast<std::size_t>(class_count);
}

std::vector<float> readout_features(vvm::Model& model, const vvm::TaskSample& sample,
                                    const vvm::TaskConfig& task_config, ReadoutSource source,
                                    std::mt19937& rng, std::size_t base_clock) {
    if (source == ReadoutSource::Input) {
        return sample.input;
    }

    std::vector<float> state = vvm::neutral_state(model.config().state_dim);
    for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
        static_cast<void>(model.tick(state, rng, base_clock + frame, sample.input));
    }
    return state;
}

vvm::ReadoutMetrics evaluate_readout(vvm::Model& model, const vvm::LinearReadout& readout,
                                     std::span<const vvm::TaskSample> samples,
                                     const vvm::TaskConfig& task_config, ReadoutSource source,
                                     std::uint32_t seed) {
    std::mt19937 rng(seed);
    vvm::ReadoutMetrics metrics{};
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const vvm::TaskSample& sample = samples[i];
        if (sample.label < 0) {
            continue;
        }
        const std::vector<float> features = readout_features(
            model, sample, task_config, source, rng, i * task_config.frames_per_sample);
        metrics.loss += readout.loss_one(features, sample.label);
        metrics.accuracy += readout.predict(features) == sample.label ? 1.0F : 0.0F;
        ++metrics.samples;
    }

    if (metrics.samples > 0U) {
        metrics.loss /= static_cast<float>(metrics.samples);
        metrics.accuracy /= static_cast<float>(metrics.samples);
    }
    return metrics;
}

vvm::ReadoutMetrics evaluate_mlp_readout(const vvm::MlpReadout& readout,
                                         std::span<const vvm::TaskSample> samples) {
    vvm::ReadoutMetrics metrics{};
    for (const vvm::TaskSample& sample : samples) {
        if (sample.label < 0) {
            continue;
        }
        metrics.loss += readout.loss_one(sample.input, sample.label);
        metrics.accuracy += readout.predict(sample.input) == sample.label ? 1.0F : 0.0F;
        ++metrics.samples;
    }

    if (metrics.samples > 0U) {
        metrics.loss /= static_cast<float>(metrics.samples);
        metrics.accuracy /= static_cast<float>(metrics.samples);
    }
    return metrics;
}

struct AutoencoderMetrics {
    float loss = 0.0F;
    float image_loss = 0.0F;
    std::size_t samples = 0;
};

AutoencoderMetrics evaluate_autoencoder(const vvm::MlpAutoencoder& autoencoder,
                                        std::span<const vvm::TaskSample> samples) {
    AutoencoderMetrics metrics{};
    for (const vvm::TaskSample& sample : samples) {
        const std::vector<float> predicted = autoencoder.predict(sample.input);
        metrics.loss += autoencoder.loss_one(sample.input, sample.input);

        const std::size_t image_dims =
            sample.class_offset > 0U ? sample.class_offset : sample.input.size();
        float image_loss = 0.0F;
        for (std::size_t i = 0; i < image_dims; ++i) {
            const float error = predicted[i] - sample.input[i];
            image_loss += error * error;
        }
        metrics.image_loss += image_loss / static_cast<float>(image_dims);
        ++metrics.samples;
    }

    if (metrics.samples > 0U) {
        metrics.loss /= static_cast<float>(metrics.samples);
        metrics.image_loss /= static_cast<float>(metrics.samples);
    }
    return metrics;
}

float span_delta_l2(std::span<const float> before, std::span<const float> after) {
    if (before.size() != after.size()) {
        throw std::invalid_argument("span_delta_l2 requires equal sizes");
    }

    float sum = 0.0F;
    for (std::size_t i = 0; i < before.size(); ++i) {
        const float delta = after[i] - before[i];
        sum += delta * delta;
    }
    return std::sqrt(sum);
}

int run_headless(const vvm::Config& config) {
    vvm::Model model(config);
    const std::vector<float> initial_state = model.seeded_state();
    const vvm::RunResult result = model.run(initial_state);

    std::cout << "steps=" << config.steps << " state_dim=" << config.state_dim
              << " ops=" << config.num_ops << " candidates=" << config.candidate_count
              << " activation=" << activation_name(config.activation)
              << " params=" << model.parameter_count() << " param_bytes=" << model.parameter_bytes()
              << '\n';

    for (std::size_t i = 0; i < result.trace.size(); ++i) {
        const vvm::StepTrace& trace = result.trace[i];
        std::cout << "step " << i << " max_score=" << trace.retrieval.max_score
                  << " state_norm=" << trace.state_norm
                  << " activation_mean=" << trace.activation_mean
                  << " prediction_error=" << trace.prediction_error
                  << " curiosity_reward=" << trace.curiosity_reward
                  << " chosen_op=" << trace.retrieval.chosen_index
                  << " chosen_score=" << trace.retrieval.chosen_score << '\n';
    }
    return 0;
}

int run_task_training(const vvm::Config& config, const vvm::TaskConfig& task_config,
                      const CliOptions& cli_options) {
    vvm::Model model(config);
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);
    const std::size_t epochs = cli_options.epochs;
    const std::vector<float> initial_bank(model.op_bank().begin(), model.op_bank().end());

    std::cout << "task=" << vvm::task_name(task_config.task) << " epochs=" << epochs
              << " train_samples=" << dataset.train.size()
              << " test_samples=" << dataset.test.size()
              << " activation=" << activation_name(config.activation)
              << " retrieval_temperature=" << config.retrieval_temperature
              << " sample_candidates=" << config.sample_candidate_count
              << " vectors=" << vector_range_name(task_config.vector_range)
              << " sample_frames=" << task_config.frames_per_sample
              << " idle_frames=" << task_config.idle_frames_between_samples
              << " window=" << task_config.window_size
              << " train_interval=" << task_config.train_interval
              << " class_start_frame=" << task_config.class_start_frame
              << " class_ramp_frames=" << task_config.class_ramp_frames
              << " class_registers=" << task_config.class_registers
              << " lr=" << task_config.learning_rate
              << " lr_decay=" << task_config.learning_rate_decay
              << " momentum=" << task_config.momentum
              << " rejection_scale=" << task_config.rejection_scale
              << " rejection_threshold=" << task_config.rejection_threshold
              << " rejection_decay=" << task_config.rejection_decay
              << " rejection_overuse_scale=" << task_config.rejection_overuse_scale
              << " affinity_retain_scale=" << task_config.affinity_retain_scale
              << " affinity_retain_threshold=" << task_config.affinity_retain_threshold
              << " affinity_retain_underuse_scale=" << task_config.affinity_retain_underuse_scale
              << " op_anchor_scale=" << task_config.op_anchor_scale
              << " anchor_to_best=" << (cli_options.anchor_to_best ? 1 : 0)
              << " world_loss_weight=" << task_config.world_loss_weight
              << " class_value_scale=" << task_config.class_value_scale
              << " class_loss_weight=" << task_config.class_loss_weight
              << " restore_best=" << (cli_options.restore_best ? 1 : 0)
              << " bptt=" << (task_config.backprop_through_state ? 1 : 0)
              << " params=" << model.parameter_count() << '\n';

    float best_accuracy = -1.0F;
    std::size_t best_accuracy_epoch = 0;
    float best_balanced_accuracy = -1.0F;
    std::size_t best_balanced_accuracy_epoch = 0;
    float best_loss = std::numeric_limits<float>::infinity();
    std::size_t best_loss_epoch = 0;
    std::vector<float> best_bank;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const std::vector<float> epoch_bank_before(model.op_bank().begin(), model.op_bank().end());
        std::span<const float> op_anchor = {};
        if (task_config.op_anchor_scale > 0.0F) {
            if (cli_options.anchor_to_best && !best_bank.empty()) {
                op_anchor = best_bank;
            } else if (!cli_options.anchor_to_best) {
                op_anchor = initial_bank;
            }
        }
        const vvm::LossPoint loss = vvm::train_task_epoch(model, dataset.train, dataset.test,
                                                          task_config, epoch, op_anchor);
        const float effective_lr =
            task_config.learning_rate *
            std::pow(task_config.learning_rate_decay, static_cast<float>(epoch));
        const float bank_delta_l2 = span_delta_l2(epoch_bank_before, model.op_bank());
        const float bank_from_init_l2 = span_delta_l2(initial_bank, model.op_bank());
        std::cout << "epoch " << std::setw(4) << epoch << " train_loss=" << loss.train_loss
                  << " self_loss=" << loss.self_loss << " test_loss=" << loss.test_loss
                  << " effective_lr=" << effective_lr;
        bool saw_new_best = false;
        if (loss.accuracy_samples > 0U) {
            std::cout << " test_nonclass_loss=" << loss.test_nonclass_loss
                      << " test_class_loss=" << loss.test_class_loss
                      << " test_class_ce=" << loss.test_class_cross_entropy;
            if (loss.test_accuracy > best_accuracy) {
                best_accuracy = loss.test_accuracy;
                best_accuracy_epoch = epoch;
            }
            if (loss.test_balanced_accuracy > best_balanced_accuracy) {
                best_balanced_accuracy = loss.test_balanced_accuracy;
                best_balanced_accuracy_epoch = epoch;
                saw_new_best = true;
            }
            std::cout << " test_accuracy=" << (100.0F * loss.test_accuracy) << "%";
            std::cout << " best_accuracy=" << (100.0F * best_accuracy) << "%" << "@"
                      << best_accuracy_epoch;
            std::cout << " balanced_accuracy=" << (100.0F * loss.test_balanced_accuracy) << "%"
                      << " best_balanced=" << (100.0F * best_balanced_accuracy) << "%" << "@"
                      << best_balanced_accuracy_epoch;
            std::cout << " class_margin=" << loss.mean_class_margin;
            print_counts("labels", loss.label_counts);
            print_counts("preds", loss.prediction_counts);
            print_recalls(loss.correct_counts, loss.label_counts);
            if (loss.class_route_purity > 0.0F) {
                std::cout << " route_purity=" << loss.class_route_purity;
            }
        }
        if (loss.accuracy_samples == 0U && loss.test_loss < best_loss) {
            best_loss = loss.test_loss;
            best_loss_epoch = epoch;
            saw_new_best = true;
        }
        if ((cli_options.restore_best || cli_options.anchor_to_best) && saw_new_best) {
            best_bank.assign(model.op_bank().begin(), model.op_bank().end());
        }
        std::cout << " heat_l2=" << (loss.state_heat_l2 + loss.op_heat_l2)
                  << " learn_l2=" << loss.learning_update_l2 << " bank_delta_l2=" << bank_delta_l2
                  << " bank_from_init_l2=" << bank_from_init_l2
                  << " max_op_heat=" << loss.max_op_heat_index << ":" << loss.max_op_heat_l2
                  << " max_op_train=" << loss.max_op_train_index << ":" << loss.max_op_train_l2
                  << " selected_ops=" << loss.selected_ops << "/" << config.num_ops
                  << " max_op_select=" << loss.max_op_selections
                  << " op_entropy=" << loss.op_selection_entropy
                  << " mean_rank=" << loss.mean_chosen_rank
                  << " mean_prob=" << loss.mean_chosen_prob
                  << " top_gap=" << loss.mean_top_score_gap
                  << " chosen_gap=" << loss.mean_chosen_score_gap
                  << " candidate_entropy=" << loss.mean_candidate_entropy;
        print_top_counts("top_select", loss.op_selection_counts);
        print_top_floats("top_train", loss.op_train_l2_by_op);
        print_top_floats("top_heat", loss.op_heat_l2_by_op);
        print_class_top_counts(loss.class_op_selection_counts, loss.label_counts.size(),
                               config.num_ops);
        std::cout << '\n';
    }
    if (cli_options.restore_best && !best_bank.empty()) {
        model.replace_op_bank(best_bank);
        std::cout << "restored_best=";
        if (best_balanced_accuracy >= 0.0F) {
            std::cout << "balanced_accuracy:" << (100.0F * best_balanced_accuracy) << "%@"
                      << best_balanced_accuracy_epoch;
        } else {
            std::cout << "test_loss:" << best_loss << "@" << best_loss_epoch;
        }
        std::cout << '\n';
    }
    return 0;
}

int run_readout_training(const vvm::Config& config, const vvm::TaskConfig& task_config,
                         const CliOptions& cli_options) {
    vvm::Model model(config);
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);
    const std::size_t class_count = infer_class_count(dataset.train);
    const std::size_t input_dim =
        cli_options.readout_source == ReadoutSource::Input ? config.state_dim : config.state_dim;

    vvm::LinearReadout readout(vvm::ReadoutConfig{
        .input_dim = input_dim,
        .class_count = class_count,
        .learning_rate = cli_options.readout_learning_rate,
    });

    std::cout << "readout task=" << vvm::task_name(task_config.task)
              << " source=" << readout_source_name(cli_options.readout_source)
              << " epochs=" << cli_options.epochs << " train_samples=" << dataset.train.size()
              << " test_samples=" << dataset.test.size() << " classes=" << class_count
              << " readout_lr=" << cli_options.readout_learning_rate
              << " sample_frames=" << task_config.frames_per_sample
              << " core_params=" << model.parameter_count()
              << " readout_params=" << readout.parameter_count()
              << " total_params=" << (model.parameter_count() + readout.parameter_count()) << '\n';

    std::vector<std::size_t> order(dataset.train.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    for (std::size_t epoch = 0; epoch < cli_options.epochs; ++epoch) {
        std::mt19937 shuffle_rng(task_config.seed ^ static_cast<std::uint32_t>(epoch));
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        float train_loss = 0.0F;
        float train_accuracy = 0.0F;
        std::size_t trained = 0;
        std::mt19937 feature_rng(task_config.seed ^ 0xA53A9U ^ static_cast<std::uint32_t>(epoch));
        for (std::size_t order_index = 0; order_index < order.size(); ++order_index) {
            const vvm::TaskSample& sample = dataset.train[order[order_index]];
            if (sample.label < 0) {
                continue;
            }
            const std::size_t clock =
                ((epoch * order.size()) + order_index) * task_config.frames_per_sample;
            const std::vector<float> features = readout_features(
                model, sample, task_config, cli_options.readout_source, feature_rng, clock);
            train_loss += readout.train_one(features, sample.label);
            train_accuracy += readout.predict(features) == sample.label ? 1.0F : 0.0F;
            ++trained;
        }

        if (trained > 0U) {
            train_loss /= static_cast<float>(trained);
            train_accuracy /= static_cast<float>(trained);
        }

        const vvm::ReadoutMetrics test =
            evaluate_readout(model, readout, dataset.test, task_config, cli_options.readout_source,
                             task_config.seed ^ 0x7E57U ^ static_cast<std::uint32_t>(epoch));

        std::cout << "epoch " << std::setw(4) << epoch << " train_loss=" << train_loss
                  << " train_accuracy=" << (100.0F * train_accuracy) << "%"
                  << " test_loss=" << test.loss << " test_accuracy=" << (100.0F * test.accuracy)
                  << "%" << " samples=" << test.samples << '\n';
    }

    return 0;
}

int run_mlp_training(const vvm::Config& config, const vvm::TaskConfig& task_config,
                     const CliOptions& cli_options) {
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);
    const std::size_t class_count = infer_class_count(dataset.train);

    vvm::MlpReadout readout(vvm::ReadoutConfig{
        .input_dim = config.state_dim,
        .hidden_dim = cli_options.hidden_dim,
        .class_count = class_count,
        .learning_rate = cli_options.readout_learning_rate,
    });

    std::cout << "mlp task=" << vvm::task_name(task_config.task) << " epochs=" << cli_options.epochs
              << " train_samples=" << dataset.train.size()
              << " test_samples=" << dataset.test.size() << " input_dim=" << config.state_dim
              << " hidden=" << cli_options.hidden_dim << " classes=" << class_count
              << " lr=" << cli_options.readout_learning_rate
              << " params=" << readout.parameter_count() << '\n';

    std::vector<std::size_t> order(dataset.train.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    for (std::size_t epoch = 0; epoch < cli_options.epochs; ++epoch) {
        std::mt19937 shuffle_rng(task_config.seed ^ static_cast<std::uint32_t>(epoch));
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        float train_loss = 0.0F;
        float train_accuracy = 0.0F;
        std::size_t trained = 0;
        for (const std::size_t sample_index : order) {
            const vvm::TaskSample& sample = dataset.train[sample_index];
            if (sample.label < 0) {
                continue;
            }
            train_loss += readout.train_one(sample.input, sample.label);
            train_accuracy += readout.predict(sample.input) == sample.label ? 1.0F : 0.0F;
            ++trained;
        }

        if (trained > 0U) {
            train_loss /= static_cast<float>(trained);
            train_accuracy /= static_cast<float>(trained);
        }

        const vvm::ReadoutMetrics test = evaluate_mlp_readout(readout, dataset.test);
        std::cout << "epoch " << std::setw(4) << epoch << " train_loss=" << train_loss
                  << " train_accuracy=" << (100.0F * train_accuracy) << "%"
                  << " test_loss=" << test.loss << " test_accuracy=" << (100.0F * test.accuracy)
                  << "%" << " samples=" << test.samples << '\n';
    }

    return 0;
}

int run_autoencoder_training(const vvm::Config& config, const vvm::TaskConfig& task_config,
                             const CliOptions& cli_options) {
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);

    vvm::MlpAutoencoder autoencoder(vvm::ReadoutConfig{
        .input_dim = config.state_dim,
        .hidden_dim = cli_options.hidden_dim,
        .learning_rate = cli_options.readout_learning_rate,
    });

    std::cout << "autoencoder task=" << vvm::task_name(task_config.task)
              << " epochs=" << cli_options.epochs << " train_samples=" << dataset.train.size()
              << " test_samples=" << dataset.test.size() << " input_dim=" << config.state_dim
              << " hidden=" << cli_options.hidden_dim << " lr=" << cli_options.readout_learning_rate
              << " params=" << autoencoder.parameter_count() << '\n';

    std::vector<std::size_t> order(dataset.train.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    for (std::size_t epoch = 0; epoch < cli_options.epochs; ++epoch) {
        std::mt19937 shuffle_rng(task_config.seed ^ static_cast<std::uint32_t>(epoch));
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        float train_loss = 0.0F;
        std::size_t trained = 0;
        for (const std::size_t sample_index : order) {
            const vvm::TaskSample& sample = dataset.train[sample_index];
            train_loss += autoencoder.train_one(sample.input, sample.input);
            ++trained;
        }

        if (trained > 0U) {
            train_loss /= static_cast<float>(trained);
        }

        const AutoencoderMetrics test = evaluate_autoencoder(autoencoder, dataset.test);
        std::cout << "epoch " << std::setw(4) << epoch << " train_loss=" << train_loss
                  << " test_loss=" << test.loss << " test_image_loss=" << test.image_loss
                  << " samples=" << test.samples << '\n';
    }

    return 0;
}

float reconstruction_white_point(const vvm::TaskSample& sample) {
    const std::size_t image_dims = std::min<std::size_t>(784U, sample.input.size());
    float white = 0.0F;
    for (std::size_t i = 0; i < image_dims; ++i) {
        white = std::max(white, sample.input[i]);
    }
    return std::max(white, 1.0e-6F);
}

unsigned char reconstruction_pixel(float value, float white_point) {
    const float normalized = std::clamp(value / white_point, 0.0F, 1.0F);
    return static_cast<unsigned char>(std::lround(normalized * 255.0F));
}

std::vector<unsigned char> reconstruction_tile(std::span<const float> values, float white_point) {
    constexpr std::size_t kImageDims = 784;
    if (values.size() < kImageDims) {
        throw std::invalid_argument("reconstruction vector must have at least 784 dims");
    }

    std::vector<unsigned char> tile(kImageDims, 0U);
    for (std::size_t i = 0; i < kImageDims; ++i) {
        tile[i] = reconstruction_pixel(values[i], white_point);
    }
    return tile;
}

void write_pgm(const std::filesystem::path& path, std::size_t width, std::size_t height,
               std::span<const unsigned char> pixels) {
    if (pixels.size() != width * height) {
        throw std::invalid_argument("PGM pixel count mismatch");
    }

    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open " + path.string());
    }
    out << "P5\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()),
              static_cast<std::streamsize>(pixels.size()));
}

void blit_tile(std::vector<unsigned char>& canvas, std::size_t canvas_width, std::size_t x,
               std::size_t y, std::span<const unsigned char> tile) {
    constexpr std::size_t kTileSize = 28;
    if (tile.size() != kTileSize * kTileSize) {
        throw std::invalid_argument("tile size mismatch");
    }
    for (std::size_t row = 0; row < kTileSize; ++row) {
        for (std::size_t col = 0; col < kTileSize; ++col) {
            canvas[((y + row) * canvas_width) + x + col] = tile[(row * kTileSize) + col];
        }
    }
}

std::vector<std::size_t>
select_binary_reconstruction_samples(std::span<const vvm::TaskSample> samples,
                                     std::size_t per_class) {
    std::vector<std::size_t> selected;
    std::size_t zeros = 0;
    std::size_t ones = 0;
    for (std::size_t i = 0; i < samples.size() && (zeros < per_class || ones < per_class); ++i) {
        if (samples[i].label == 0 && zeros < per_class) {
            selected.push_back(i);
            ++zeros;
        } else if (samples[i].label == 1 && ones < per_class) {
            selected.push_back(i);
            ++ones;
        }
    }
    if (selected.empty()) {
        throw std::runtime_error("no labeled samples available for reconstruction export");
    }
    return selected;
}

int run_reconstruction_export(const vvm::Config& config, const vvm::TaskConfig& task_config,
                              const CliOptions& cli_options) {
    vvm::Model model(config);
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);

    vvm::MlpAutoencoder autoencoder(vvm::ReadoutConfig{
        .input_dim = config.state_dim,
        .hidden_dim = cli_options.hidden_dim,
        .learning_rate = cli_options.readout_learning_rate,
    });

    std::vector<std::size_t> order(dataset.train.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    for (std::size_t epoch = 0; epoch < cli_options.epochs; ++epoch) {
        std::mt19937 shuffle_rng(task_config.seed ^ static_cast<std::uint32_t>(epoch));
        std::shuffle(order.begin(), order.end(), shuffle_rng);
        for (const std::size_t sample_index : order) {
            const vvm::TaskSample& sample = dataset.train[sample_index];
            static_cast<void>(autoencoder.train_one(sample.input, sample.input));
        }
    }

    float best_balanced_accuracy = -1.0F;
    std::vector<float> best_bank;
    const std::vector<float> initial_bank(model.op_bank().begin(), model.op_bank().end());
    for (std::size_t epoch = 0; epoch < cli_options.epochs; ++epoch) {
        std::span<const float> op_anchor = {};
        if (task_config.op_anchor_scale > 0.0F) {
            if (cli_options.anchor_to_best && !best_bank.empty()) {
                op_anchor = best_bank;
            } else if (!cli_options.anchor_to_best) {
                op_anchor = initial_bank;
            }
        }
        const vvm::LossPoint loss = vvm::train_task_epoch(model, dataset.train, dataset.test,
                                                          task_config, epoch, op_anchor);
        if (loss.accuracy_samples > 0U && loss.test_balanced_accuracy > best_balanced_accuracy) {
            best_balanced_accuracy = loss.test_balanced_accuracy;
            if (cli_options.restore_best || cli_options.anchor_to_best) {
                best_bank.assign(model.op_bank().begin(), model.op_bank().end());
            }
        }
    }
    if (cli_options.restore_best && !best_bank.empty()) {
        model.replace_op_bank(best_bank);
    }

    const std::filesystem::path output_root(cli_options.output_dir);
    std::filesystem::create_directories(output_root);
    const std::vector<std::size_t> selected =
        select_binary_reconstruction_samples(dataset.test, 5U);

    constexpr std::size_t kTileSize = 28;
    constexpr std::size_t kGap = 2;
    const std::size_t columns = selected.size();
    const std::size_t rows = 3;
    const std::size_t canvas_width = (columns * kTileSize) + ((columns - 1U) * kGap);
    const std::size_t canvas_height = (rows * kTileSize) + ((rows - 1U) * kGap);
    std::vector<unsigned char> contact(canvas_width * canvas_height, 24U);

    for (std::size_t column = 0; column < selected.size(); ++column) {
        const vvm::TaskSample& sample = dataset.test[selected[column]];
        const float white_point = reconstruction_white_point(sample);
        const std::vector<unsigned char> original = reconstruction_tile(sample.input, white_point);
        const std::vector<float> autoencoder_output = autoencoder.predict(sample.input);
        const std::vector<unsigned char> autoencoder_tile =
            reconstruction_tile(autoencoder_output, white_point);

        std::vector<float> state = vvm::neutral_state(model.config().state_dim);
        for (std::size_t frame = 0; frame < task_config.frames_per_sample; ++frame) {
            state = model.predict_next(state, sample.input);
        }
        const std::vector<unsigned char> vvm_tile = reconstruction_tile(state, white_point);

        const std::size_t x = column * (kTileSize + kGap);
        blit_tile(contact, canvas_width, x, 0U, original);
        blit_tile(contact, canvas_width, x, kTileSize + kGap, autoencoder_tile);
        blit_tile(contact, canvas_width, x, (2U * (kTileSize + kGap)), vvm_tile);

        const std::string stem =
            std::to_string(column) + "_label" + std::to_string(sample.label) + ".pgm";
        write_pgm(output_root / ("original_" + stem), kTileSize, kTileSize, original);
        write_pgm(output_root / ("autoencoder_" + stem), kTileSize, kTileSize, autoencoder_tile);
        write_pgm(output_root / ("vvm_" + stem), kTileSize, kTileSize, vvm_tile);
    }

    write_pgm(output_root / "mnist01_reconstruction_contact.pgm", canvas_width, canvas_height,
              contact);

    std::cout << "reconstruction_export output_dir=" << output_root
              << " samples=" << selected.size() << " rows=original,autoencoder,vvm"
              << " autoencoder_params=" << autoencoder.parameter_count()
              << " vvm_params=" << model.parameter_count();
    if (best_balanced_accuracy >= 0.0F) {
        std::cout << " vvm_best_balanced=" << (100.0F * best_balanced_accuracy) << "%";
    }
    std::cout << '\n';
    return 0;
}

int run_task_benchmarks(vvm::Config config, vvm::TaskConfig base_task_config, std::size_t epochs) {
    struct BenchTask {
        vvm::TaskKind task = vvm::TaskKind::CopyInput;
        std::size_t idle_frames = 0;
    };

    const BenchTask tasks[] = {
        BenchTask{.task = vvm::TaskKind::CopyInput, .idle_frames = 0},
        BenchTask{.task = vvm::TaskKind::DelayedCopy, .idle_frames = 8},
        BenchTask{.task = vvm::TaskKind::AlternatingBit, .idle_frames = 0},
        BenchTask{.task = vvm::TaskKind::Xor, .idle_frames = 0},
        BenchTask{.task = vvm::TaskKind::SineNext, .idle_frames = 0},
    };

    std::cout << "task_bench" << " epochs=" << epochs << " state_dim=" << config.state_dim
              << " ops=" << config.num_ops << " candidates=" << config.candidate_count
              << " activation=" << activation_name(config.activation)
              << " vectors=" << vector_range_name(base_task_config.vector_range)
              << " train_samples=" << base_task_config.train_samples
              << " test_samples=" << base_task_config.test_samples
              << " sample_frames=" << base_task_config.frames_per_sample
              << " window=" << base_task_config.window_size << '\n';

    for (const BenchTask& bench_task : tasks) {
        vvm::TaskConfig task_config = base_task_config;
        task_config.task = bench_task.task;
        task_config.idle_frames_between_samples = bench_task.idle_frames;

        vvm::Model model(config);
        const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);
        const vvm::EvalMetrics initial_metrics =
            vvm::evaluate_task_metrics(model, dataset.test, task_config);

        const auto begin = std::chrono::steady_clock::now();
        vvm::LossPoint loss{};
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            loss = vvm::train_task_epoch(model, dataset.train, dataset.test, task_config, epoch);
        }
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - begin).count();
        const std::size_t ticks_per_epoch =
            task_config.train_samples *
            (task_config.frames_per_sample + task_config.idle_frames_between_samples);
        const std::size_t train_ticks = ticks_per_epoch * epochs;
        const double tick_rate = seconds > 0.0 ? static_cast<double>(train_ticks) / seconds : 0.0;
        const double epoch_rate = seconds > 0.0 ? static_cast<double>(epochs) / seconds : 0.0;
        const float improvement = initial_metrics.loss - loss.test_loss;

        std::cout << "task=" << vvm::task_name(task_config.task)
                  << " initial_test=" << initial_metrics.loss << " final_train=" << loss.train_loss
                  << " final_self=" << loss.self_loss << " final_test=" << loss.test_loss
                  << " improvement=" << improvement;
        if (loss.accuracy_samples > 0U) {
            std::cout << " final_accuracy=" << (100.0F * loss.test_accuracy) << "%";
        }
        std::cout << " heat_l2=" << (loss.state_heat_l2 + loss.op_heat_l2)
                  << " learn_l2=" << loss.learning_update_l2
                  << " max_op_heat=" << loss.max_op_heat_index << ":" << loss.max_op_heat_l2
                  << " max_op_train=" << loss.max_op_train_index << ":" << loss.max_op_train_l2
                  << " selected_ops=" << loss.selected_ops << "/" << config.num_ops
                  << " max_op_select=" << loss.max_op_selections
                  << " op_entropy=" << loss.op_selection_entropy << " seconds=" << seconds
                  << " train_ticks=" << train_ticks << " tick_rate=" << tick_rate
                  << " epoch_rate=" << epoch_rate << '\n';
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    try {
        vvm::Config config{};
        vvm::TaskConfig task_config{};
        CliOptions cli_options{};
        const std::string_view command(argv[1]);
        const std::span<char*> options(argv + 2, static_cast<std::size_t>(argc - 2));
        if (!parse_options(options, config, task_config, cli_options)) {
            print_usage();
            return 2;
        }

        if (command == "smoke" || command == "run") {
            return run_headless(config);
        }

        if (command == "train-task") {
            return run_task_training(config, task_config, cli_options);
        }

        if (command == "train-readout") {
            return run_readout_training(config, task_config, cli_options);
        }

        if (command == "train-mlp") {
            return run_mlp_training(config, task_config, cli_options);
        }

        if (command == "train-autoencoder") {
            return run_autoencoder_training(config, task_config, cli_options);
        }

        if (command == "export-reconstructions") {
            return run_reconstruction_export(config, task_config, cli_options);
        }

        if (command == "bench-tasks") {
            return run_task_benchmarks(config, task_config, cli_options.epochs);
        }

        if (command == "visualize") {
#ifdef VVM_WITH_SDL3
            return vvm::run_visualizer(config);
#else
            std::cerr << "visualizer was not built. Reconfigure with cmake --preset dev-sdl3.\n";
            return 2;
#endif
        }

        if (command == "visualize-train") {
#ifdef VVM_WITH_SDL3
            return vvm::run_training_visualizer(config, task_config, cli_options.epochs);
#else
            std::cerr << "visualizer was not built. Reconfigure with cmake --preset dev-sdl3.\n";
            return 2;
#endif
        }

        print_usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
