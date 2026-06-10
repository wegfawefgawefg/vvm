#include "vvm/model.hpp"
#include "vvm/sdl_text.hpp"
#include "vvm/toy_training.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <span>
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

void draw_loss_graph(SDL_Renderer* renderer, const Model& model, const ToyTaskConfig& task_config,
                     std::span<const LossPoint> history, std::size_t target_epochs, int width,
                     int height) {
    SDL_SetRenderDrawColor(renderer, 10, 12, 16, 255);
    SDL_RenderClear(renderer);

    const float left = 56.0F;
    const float top = 72.0F;
    const float right = static_cast<float>(width) - 28.0F;
    const float bottom = static_cast<float>(height) - 48.0F;
    const float graph_width = std::max(1.0F, right - left);
    const float graph_height = std::max(1.0F, bottom - top);

    SDL_SetRenderDrawColor(renderer, 35, 40, 48, 255);
    SDL_FRect graph_bg{left, top, graph_width, graph_height};
    SDL_RenderFillRect(renderer, &graph_bg);

    SDL_SetRenderDrawColor(renderer, 80, 88, 104, 255);
    SDL_RenderLine(renderer, left, bottom, right, bottom);
    SDL_RenderLine(renderer, left, top, left, bottom);

    float max_loss = 1.0e-6F;
    for (const LossPoint& point : history) {
        max_loss =
            std::max(max_loss, std::max({point.train_loss, point.self_loss, point.test_loss}));
    }
    max_loss *= 1.05F;

    auto point_for = [&](std::size_t i, float loss) {
        const std::size_t target_epoch_count = std::max<std::size_t>(1U, target_epochs);
        const float denom = std::max(1.0F, static_cast<float>(target_epoch_count - 1U));
        const float x = left + (static_cast<float>(i) / denom) * graph_width;
        const float y = bottom - std::clamp(loss / max_loss, 0.0F, 1.0F) * graph_height;
        return SDL_FPoint{x, y};
    };

    for (std::size_t i = 1; i < history.size(); ++i) {
        const SDL_FPoint train_prev = point_for(i - 1U, history[i - 1U].train_loss);
        const SDL_FPoint train_now = point_for(i, history[i].train_loss);
        SDL_SetRenderDrawColor(renderer, 90, 210, 245, 255);
        SDL_RenderLine(renderer, train_prev.x, train_prev.y, train_now.x, train_now.y);

        const SDL_FPoint test_prev = point_for(i - 1U, history[i - 1U].test_loss);
        const SDL_FPoint test_now = point_for(i, history[i].test_loss);
        SDL_SetRenderDrawColor(renderer, 245, 185, 80, 255);
        SDL_RenderLine(renderer, test_prev.x, test_prev.y, test_now.x, test_now.y);

        const SDL_FPoint self_prev = point_for(i - 1U, history[i - 1U].self_loss);
        const SDL_FPoint self_now = point_for(i, history[i].self_loss);
        SDL_SetRenderDrawColor(renderer, 145, 235, 135, 255);
        SDL_RenderLine(renderer, self_prev.x, self_prev.y, self_now.x, self_now.y);
    }

    const LossPoint latest = history.empty() ? LossPoint{} : history.back();
    const double mib = static_cast<double>(model.parameter_bytes()) / (1024.0 * 1024.0);

    char overlay[640];
    std::snprintf(overlay, sizeof(overlay),
                  "vvm toy training  epoch=%zu/%zu  params=%zu  %.2f MiB\n"
                  "train_loss=%.6f  self_loss=%.6f  test_loss=%.6f  max_loss=%.6f\n"
                  "samples train=%zu test=%zu  sample_frames=%zu  idle_frames=%zu  window=%zu  "
                  "lr=%.4f",
                  history.size(), target_epochs, model.parameter_count(), mib,
                  static_cast<double>(latest.train_loss), static_cast<double>(latest.self_loss),
                  static_cast<double>(latest.test_loss), static_cast<double>(max_loss),
                  task_config.train_samples, task_config.test_samples,
                  task_config.frames_per_sample, task_config.idle_frames_between_samples,
                  task_config.window_size, static_cast<double>(task_config.learning_rate));

    draw_text(renderer, 20, overlay, 14.0F, 14.0F, SDL_Color{235, 240, 245, 255});
    draw_text(renderer, 16, "train", right - 140.0F, top + 12.0F, SDL_Color{90, 210, 245, 255});
    draw_text(renderer, 16, "test", right - 140.0F, top + 36.0F, SDL_Color{245, 185, 80, 255});
    draw_text(renderer, 16, "self", right - 140.0F, top + 60.0F, SDL_Color{145, 235, 135, 255});
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

int run_training_visualizer(const Config& config, const ToyTaskConfig& task_config,
                            std::size_t epochs) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw_sdl_error("SDL_Init failed");
    }

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    if (!SDL_CreateWindowAndRenderer("vvm training", 1080, 680, SDL_WINDOW_RESIZABLE, &window,
                                     &renderer)) {
        SDL_Quit();
        throw_sdl_error("SDL_CreateWindowAndRenderer failed");
    }
    init_text_subsystem();

    Model model(config);
    const ToyDataset dataset = make_toy_dataset(config, task_config);
    std::vector<LossPoint> history;
    history.reserve(epochs);

    bool running = true;
    while (running) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }

        if (history.size() < epochs) {
            history.push_back(
                train_toy_epoch(model, dataset.train, dataset.test, task_config, history.size()));
        }

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        draw_loss_graph(renderer, model, task_config, history, epochs, width, height);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    shutdown_text_subsystem();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    if (!history.empty()) {
        const LossPoint latest = history.back();
        std::cout << "trained " << history.size() << " epochs train_loss=" << latest.train_loss
                  << " self_loss=" << latest.self_loss << " test_loss=" << latest.test_loss << '\n';
    }
    return 0;
}

} // namespace vvm
