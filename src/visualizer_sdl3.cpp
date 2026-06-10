#include "vvm/model.hpp"
#include "vvm/sdl_text.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vvm {
namespace {

void throw_sdl_error(const char* message) {
    throw std::runtime_error(std::string(message) + ": " + SDL_GetError());
}

void draw_trace(SDL_Renderer* renderer, const Model& model, const RunResult& result, int width,
                int height) {
    SDL_SetRenderDrawColor(renderer, 10, 12, 16, 255);
    SDL_RenderClear(renderer);

    const int steps = static_cast<int>(result.trace.size());
    if (steps == 0) {
        return;
    }

    const float cell_width = static_cast<float>(width) / static_cast<float>(steps);
    const float half_height = static_cast<float>(height) * 0.5F;

    for (int i = 0; i < steps; ++i) {
        const StepTrace& trace = result.trace[static_cast<std::size_t>(i)];
        const float score = std::clamp((trace.retrieval.max_score + 1.0F) * 0.5F, 0.0F, 1.0F);
        const float activation = std::clamp(trace.activation_mean, 0.0F, 1.0F);

        SDL_FRect score_bar{
            .x = static_cast<float>(i) * cell_width,
            .y = half_height * (1.0F - score),
            .w = std::max(1.0F, cell_width - 1.0F),
            .h = half_height * score,
        };
        SDL_SetRenderDrawColor(renderer, 80, 180, 220, 255);
        SDL_RenderFillRect(renderer, &score_bar);

        SDL_FRect activation_bar{
            .x = static_cast<float>(i) * cell_width,
            .y = half_height + (half_height * (1.0F - activation)),
            .w = std::max(1.0F, cell_width - 1.0F),
            .h = half_height * activation,
        };
        SDL_SetRenderDrawColor(renderer, 230, 160, 70, 255);
        SDL_RenderFillRect(renderer, &activation_bar);
    }

    const StepTrace& latest = result.trace.back();
    const Config& config = model.config();
    const double mib = static_cast<double>(model.parameter_bytes()) / (1024.0 * 1024.0);

    char overlay[512];
    std::snprintf(
        overlay, sizeof(overlay),
        "vvm  ops=%zu  d=%zu  params=%zu  %.2f MiB\n"
        "steps=%zu  k=%zu  chosen=%zu  max=%.3f  chosen_score=%.3f\n"
        "state_norm=%.3f  act=%.3f  pred_err=%.6f  curiosity=%.6f",
        config.num_ops, config.state_dim, model.parameter_count(), mib, config.steps,
        config.candidate_count, latest.retrieval.chosen_index,
        static_cast<double>(latest.retrieval.max_score),
        static_cast<double>(latest.retrieval.chosen_score), static_cast<double>(latest.state_norm),
        static_cast<double>(latest.activation_mean), static_cast<double>(latest.prediction_error),
        static_cast<double>(latest.curiosity_reward));

    draw_text(renderer, 20, overlay, 14.0F, 14.0F, SDL_Color{235, 240, 245, 255});
}

} // namespace

int run_visualizer(const Config& config) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw_sdl_error("SDL_Init failed");
    }

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    if (!SDL_CreateWindowAndRenderer("vvm trace", 960, 540, SDL_WINDOW_RESIZABLE, &window,
                                     &renderer)) {
        SDL_Quit();
        throw_sdl_error("SDL_CreateWindowAndRenderer failed");
    }
    init_text_subsystem();

    Model model(config);
    const std::vector<float> initial_state = model.seeded_state();
    const RunResult result = model.run(initial_state);

    bool running = true;
    while (running) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        draw_trace(renderer, model, result, width, height);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    shutdown_text_subsystem();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "visualized " << result.trace.size() << " VVM steps\n";
    return 0;
}

} // namespace vvm
