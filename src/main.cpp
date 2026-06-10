#include "vvm/model.hpp"
#include "vvm/tasks.hpp"

#include <charconv>
#include <chrono>
#include <exception>
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

void print_usage() {
    std::cout << "usage:\n"
              << "  vvm smoke\n"
              << "  vvm run [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--activation deadzone] [--update-scale F] "
                 "[--state-heat F] [--op-heat F] [--heat-decay F]\n"
              << "  vvm train-task [--epochs N] [--train-samples N] [--test-samples N] "
                 "[--task copy-input|delayed-copy|alternating-bit|xor|sine-next|mnist] "
                 "[--mnist-dir PATH] [--vectors signed|nonnegative] [--sample-frames N] "
                 "[--idle-frames N] [--window N] [--lr F]\n"
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
                   std::size_t& epochs) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg(args[i]);
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
            if (!parse_size(value, epochs)) {
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
        } else if (arg == "--lr" || arg == "--learning-rate") {
            if (!parse_float(value, task_config.learning_rate)) {
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
        } else {
            std::cerr << "unknown option: " << arg << '\n';
            return false;
        }
        ++i;
    }
    return true;
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
                      std::size_t epochs) {
    vvm::Model model(config);
    const vvm::TaskDataset dataset = vvm::make_task_dataset(config, task_config);

    std::cout << "task=" << vvm::task_name(task_config.task) << " epochs=" << epochs
              << " train_samples=" << dataset.train.size()
              << " test_samples=" << dataset.test.size()
              << " activation=" << activation_name(config.activation)
              << " vectors=" << vector_range_name(task_config.vector_range)
              << " sample_frames=" << task_config.frames_per_sample
              << " idle_frames=" << task_config.idle_frames_between_samples
              << " window=" << task_config.window_size << " lr=" << task_config.learning_rate
              << " rejection_scale=" << task_config.rejection_scale
              << " rejection_threshold=" << task_config.rejection_threshold
              << " params=" << model.parameter_count() << '\n';

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const vvm::LossPoint loss =
            vvm::train_task_epoch(model, dataset.train, dataset.test, task_config, epoch);
        std::cout << "epoch " << std::setw(4) << epoch << " train_loss=" << loss.train_loss
                  << " self_loss=" << loss.self_loss << " test_loss=" << loss.test_loss;
        if (loss.accuracy_samples > 0U) {
            std::cout << " test_accuracy=" << (100.0F * loss.test_accuracy) << "%";
        }
        std::cout << '\n';
    }
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
        std::cout << " seconds=" << seconds << " train_ticks=" << train_ticks
                  << " tick_rate=" << tick_rate << " epoch_rate=" << epoch_rate << '\n';
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
        std::size_t epochs = 200;
        const std::string_view command(argv[1]);
        const std::span<char*> options(argv + 2, static_cast<std::size_t>(argc - 2));
        if (!parse_options(options, config, task_config, epochs)) {
            print_usage();
            return 2;
        }

        if (command == "smoke" || command == "run") {
            return run_headless(config);
        }

        if (command == "train-task") {
            return run_task_training(config, task_config, epochs);
        }

        if (command == "bench-tasks") {
            return run_task_benchmarks(config, task_config, epochs);
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
            return vvm::run_training_visualizer(config, task_config, epochs);
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
