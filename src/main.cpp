#include "vvm/model.hpp"

#include <charconv>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#ifdef VVM_WITH_SDL3
namespace vvm {
int run_visualizer(const Config& config);
}
#endif

namespace {

void print_usage() {
    std::cout << "usage:\n"
              << "  vvm smoke\n"
              << "  vvm run [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--update-scale F] [--state-heat F] [--op-heat F] [--heat-decay F]\n"
              << "  vvm visualize [--steps N] [--state-dim N] [--ops N] [--candidates N] "
                 "[--update-scale F] [--state-heat F] [--op-heat F] [--heat-decay F]\n";
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

bool parse_config(std::span<char*> args, vvm::Config& config) {
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
              << " ops=" << config.num_ops << " candidates=" << config.candidate_count << '\n';

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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    try {
        vvm::Config config{};
        const std::string_view command(argv[1]);
        const std::span<char*> options(argv + 2, static_cast<std::size_t>(argc - 2));
        if (!parse_config(options, config)) {
            print_usage();
            return 2;
        }

        if (command == "smoke" || command == "run") {
            return run_headless(config);
        }

        if (command == "visualize") {
#ifdef VVM_WITH_SDL3
            return vvm::run_visualizer(config);
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
