#include "nnvm/model.hpp"

#include <charconv>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#ifdef NNVM_WITH_SDL3
namespace nnvm {
int run_visualizer(const Config& config);
}
#endif

namespace {

void print_usage() {
    std::cout << "usage:\n"
              << "  nnvm smoke\n"
              << "  nnvm run [--steps N] [--state-dim N] [--ops N] [--top-k N] [--temperature F]\n"
              << "  nnvm visualize [--steps N] [--state-dim N] [--ops N] [--top-k N]\n";
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

bool parse_config(std::span<char*> args, nnvm::Config& config) {
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
        } else if (arg == "--top-k") {
            if (!parse_size(value, config.top_k)) {
                return false;
            }
        } else if (arg == "--temperature") {
            if (!parse_float(value, config.temperature)) {
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

int run_headless(const nnvm::Config& config) {
    const nnvm::Model model(config);
    const std::vector<float> initial_state = model.seeded_state();
    const nnvm::RunResult result = model.run(initial_state);

    std::cout << "steps=" << config.steps << " state_dim=" << config.state_dim
              << " ops=" << config.num_ops << " top_k=" << config.top_k << '\n';

    for (std::size_t i = 0; i < result.trace.size(); ++i) {
        const nnvm::StepTrace& trace = result.trace[i];
        std::cout << "step " << i << " max_score=" << trace.retrieval.max_score
                  << " state_norm=" << trace.state_norm << " gate_mean=" << trace.gate_mean
                  << " top_op=" << trace.retrieval.indices.front() << '\n';
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
        nnvm::Config config{};
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
#ifdef NNVM_WITH_SDL3
            return nnvm::run_visualizer(config);
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
