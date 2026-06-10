#include "vvm/model.hpp"
#include "vvm/toy_training.hpp"

#include <charconv>
#include <exception>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#ifdef VVM_WITH_SDL3
namespace vvm {
int run_visualizer(const Config& config);
int run_training_visualizer(const Config& config, const ToyTaskConfig& task_config,
                            std::size_t epochs);
} // namespace vvm
#endif

namespace {

void print_usage() {
    std::cout << "usage:\n"
              << "  vvm smoke\n"
              << "  vvm run [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--update-scale F] [--state-heat F] [--op-heat F] [--heat-decay F]\n"
              << "  vvm train-toy [--epochs N] [--train-samples N] [--test-samples N] "
                 "[--sample-frames N] [--idle-frames N] [--window N] [--lr F]\n"
              << "  vvm visualize [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--update-scale F] [--state-heat F] [--op-heat F] [--heat-decay F]\n"
              << "  vvm visualize-train [--epochs N] [--train-samples N] [--test-samples N] "
                 "[--sample-frames N] [--idle-frames N] [--window N] [--lr F]\n";
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

bool parse_options(std::span<char*> args, vvm::Config& config, vvm::ToyTaskConfig& task_config,
                   std::size_t& epochs) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg(args[i]);
        if (i + 1 >= args.size()) {
            std::cerr << "missing value for " << arg << '\n';
            return false;
        }

        const std::string_view value(args[i + 1]);
        if (arg == "--steps") {
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

int run_toy_training(const vvm::Config& config, const vvm::ToyTaskConfig& task_config,
                     std::size_t epochs) {
    vvm::Model model(config);
    const vvm::ToyDataset dataset = vvm::make_toy_dataset(config, task_config);

    std::cout << "toy=copy_input" << " epochs=" << epochs
              << " train_samples=" << dataset.train.size()
              << " test_samples=" << dataset.test.size()
              << " sample_frames=" << task_config.frames_per_sample
              << " idle_frames=" << task_config.idle_frames_between_samples
              << " window=" << task_config.window_size << " lr=" << task_config.learning_rate
              << " params=" << model.parameter_count() << '\n';

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const vvm::LossPoint loss =
            vvm::train_toy_epoch(model, dataset.train, dataset.test, task_config, epoch);
        std::cout << "epoch " << std::setw(4) << epoch << " train_loss=" << loss.train_loss
                  << " self_loss=" << loss.self_loss << " test_loss=" << loss.test_loss << '\n';
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
        vvm::ToyTaskConfig task_config{};
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

        if (command == "train-toy") {
            return run_toy_training(config, task_config, epochs);
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
