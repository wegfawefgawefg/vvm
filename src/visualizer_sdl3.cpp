#include "vvm/model.hpp"
#include "vvm/sdl_text.hpp"
#include "vvm/tasks.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_sdlrenderer3.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <imgui.h>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vvm {
namespace {

void throw_sdl_error(const char* message) {
    throw std::runtime_error(std::string(message) + ": " + SDL_GetError());
}

void throw_imgui_error(const char* message) {
    throw std::runtime_error(message);
}

void init_imgui(SDL_Window* window, SDL_Renderer* renderer) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    if (!ImGui_ImplSDL3_InitForSDLRenderer(window, renderer)) {
        ImGui::DestroyContext();
        throw_imgui_error("ImGui_ImplSDL3_InitForSDLRenderer failed");
    }
    if (!ImGui_ImplSDLRenderer3_Init(renderer)) {
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        throw_imgui_error("ImGui_ImplSDLRenderer3_Init failed");
    }
}

void shutdown_imgui() {
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

void new_imgui_frame() {
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void render_imgui(SDL_Renderer* renderer) {
    ImGui::Render();
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
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

void draw_loss_graph(SDL_Renderer* renderer, const Model& model, const TaskConfig& task_config,
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

    float max_diag = 1.0e-6F;
    for (const LossPoint& point : history) {
        max_diag = std::max(max_diag, point.state_heat_l2 + point.op_heat_l2);
        max_diag = std::max(max_diag, point.learning_update_l2);
    }
    max_diag *= 1.05F;

    auto diag_point_for = [&](std::size_t i, float value) {
        const std::size_t target_epoch_count = std::max<std::size_t>(1U, target_epochs);
        const float denom = std::max(1.0F, static_cast<float>(target_epoch_count - 1U));
        const float x = left + (static_cast<float>(i) / denom) * graph_width;
        const float y = bottom - std::clamp(value / max_diag, 0.0F, 1.0F) * graph_height;
        return SDL_FPoint{x, y};
    };

    for (std::size_t i = 1; i < history.size(); ++i) {
        const SDL_FPoint heat_prev =
            diag_point_for(i - 1U, history[i - 1U].state_heat_l2 + history[i - 1U].op_heat_l2);
        const SDL_FPoint heat_now =
            diag_point_for(i, history[i].state_heat_l2 + history[i].op_heat_l2);
        SDL_SetRenderDrawColor(renderer, 245, 95, 95, 255);
        SDL_RenderLine(renderer, heat_prev.x, heat_prev.y, heat_now.x, heat_now.y);

        const SDL_FPoint learn_prev = diag_point_for(i - 1U, history[i - 1U].learning_update_l2);
        const SDL_FPoint learn_now = diag_point_for(i, history[i].learning_update_l2);
        SDL_SetRenderDrawColor(renderer, 185, 145, 255, 255);
        SDL_RenderLine(renderer, learn_prev.x, learn_prev.y, learn_now.x, learn_now.y);
    }

    const LossPoint latest = history.empty() ? LossPoint{} : history.back();
    const double mib = static_cast<double>(model.parameter_bytes()) / (1024.0 * 1024.0);

    char overlay[640];
    std::snprintf(overlay, sizeof(overlay),
                  "vvm task training  epoch=%zu/%zu  params=%zu  %.2f MiB\n"
                  "train_loss=%.6f  self_loss=%.6f  test_loss=%.6f  max_loss=%.6f\n"
                  "heat_l2=%.6f  learn_l2=%.6f  ops=%zu/%zu  max_op=%zu  entropy=%.3f\n"
                  "samples train=%zu test=%zu  sample_frames=%zu  idle_frames=%zu  window=%zu  "
                  "lr=%.4f  reject=%.4f@%.4f",
                  history.size(), target_epochs, model.parameter_count(), mib,
                  static_cast<double>(latest.train_loss), static_cast<double>(latest.self_loss),
                  static_cast<double>(latest.test_loss), static_cast<double>(max_loss),
                  static_cast<double>(latest.state_heat_l2 + latest.op_heat_l2),
                  static_cast<double>(latest.learning_update_l2), latest.selected_ops,
                  model.config().num_ops, latest.max_op_selections,
                  static_cast<double>(latest.op_selection_entropy), task_config.train_samples,
                  task_config.test_samples, task_config.frames_per_sample,
                  task_config.idle_frames_between_samples, task_config.window_size,
                  static_cast<double>(task_config.learning_rate),
                  static_cast<double>(task_config.rejection_scale),
                  static_cast<double>(task_config.rejection_threshold));

    draw_text(renderer, 20, overlay, 14.0F, 14.0F, SDL_Color{235, 240, 245, 255});
    draw_text(renderer, 16, "train", right - 140.0F, top + 12.0F, SDL_Color{90, 210, 245, 255});
    draw_text(renderer, 16, "test", right - 140.0F, top + 36.0F, SDL_Color{245, 185, 80, 255});
    draw_text(renderer, 16, "self", right - 140.0F, top + 60.0F, SDL_Color{145, 235, 135, 255});
    draw_text(renderer, 16, "heat", right - 140.0F, top + 84.0F, SDL_Color{245, 95, 95, 255});
    draw_text(renderer, 16, "learn", right - 140.0F, top + 108.0F, SDL_Color{185, 145, 255, 255});
}

float image_white_point(std::span<const float> values) {
    const std::size_t dims = std::min<std::size_t>(784U, values.size());
    float white = 0.0F;
    for (std::size_t i = 0; i < dims; ++i) {
        white = std::max(white, std::fabs(values[i]));
    }
    return std::max(white, 1.0e-6F);
}

constexpr float kMnistDisplayWhitePoint = 0.14F;
constexpr float kStateDisplayWhitePoint = 0.08F;

std::span<const float> image_region(std::span<const float> values, std::size_t offset) {
    constexpr std::size_t kImageDims = 784;
    if (offset > values.size() || values.size() - offset < kImageDims) {
        return {};
    }
    return values.subspan(offset, kImageDims);
}

void draw_state_image(SDL_Renderer* renderer, std::span<const float> values, float x, float y,
                      float scale, float white_point, bool signed_values) {
    constexpr std::size_t kImageSize = 28;
    if (values.size() < kImageSize * kImageSize) {
        return;
    }

    for (std::size_t row = 0; row < kImageSize; ++row) {
        for (std::size_t col = 0; col < kImageSize; ++col) {
            const float raw_value = values[(row * kImageSize) + col];
            const float value = std::isfinite(raw_value) ? raw_value : 0.0F;
            if (signed_values && value < 0.0F) {
                const float intensity = std::clamp(-value / white_point, 0.0F, 1.0F);
                SDL_SetRenderDrawColor(renderer, static_cast<Uint8>(std::lround(80.0F * intensity)),
                                       static_cast<Uint8>(std::lround(150.0F * intensity)),
                                       static_cast<Uint8>(std::lround(255.0F * intensity)), 255);
            } else {
                const float intensity = std::clamp(value / white_point, 0.0F, 1.0F);
                const Uint8 channel = static_cast<Uint8>(std::lround(255.0F * intensity));
                SDL_SetRenderDrawColor(renderer, channel, channel, channel, 255);
            }
            const SDL_FRect pixel{
                .x = x + (static_cast<float>(col) * scale),
                .y = y + (static_cast<float>(row) * scale),
                .w = scale,
                .h = scale,
            };
            SDL_RenderFillRect(renderer, &pixel);
        }
    }
}

void format_clock_label(char* buffer, std::size_t size, int clock_level) {
    if (clock_level >= 0) {
        std::snprintf(buffer, size, "CLOCK x%d", clock_level + 1);
        return;
    }

    const int divisor = 1 << std::min(10, -clock_level);
    std::snprintf(buffer, size, "CLOCK 1/%dx", divisor);
}

void format_clock_detail(char* buffer, std::size_t size, int clock_level) {
    if (clock_level >= 0) {
        const int ticks = clock_level + 1;
        std::snprintf(buffer, size, "clock=%d tick%s/frame", ticks, ticks == 1 ? "" : "s");
        return;
    }

    const int divisor = 1 << std::min(10, -clock_level);
    std::snprintf(buffer, size, "clock=1 tick every %d frames", divisor);
}

void draw_probe(SDL_Renderer* renderer, const Model& model, const TaskDataset& dataset,
                std::span<const float> state, std::size_t sample_index, bool input_enabled,
                bool paused, std::size_t tick_count, int clock_level, const Tick* last_tick,
                int width, int height) {
    SDL_SetRenderDrawColor(renderer, 14, 16, 20, 255);
    SDL_RenderClear(renderer);

    if (dataset.test.empty()) {
        draw_text(renderer, 20, "no test samples", 20.0F, 20.0F, SDL_Color{240, 240, 240, 255});
        return;
    }

    const TaskSample& sample = dataset.test[sample_index % dataset.test.size()];
    const float panel_scale = std::max(
        4.0F, std::min(static_cast<float>(width) / 145.0F, static_cast<float>(height) / 58.0F));
    const float image_size = 28.0F * panel_scale;
    const float top = 150.0F;
    const float gap = 28.0F;
    const float left = 28.0F;

    const float original_white = image_white_point(sample.input);
    const float state_white = image_white_point(state);
    std::vector<float> input_view(sample.input.size(), 0.0F);
    if (input_enabled) {
        input_view = sample.input;
    }

    std::vector<float> diff(784U, 0.0F);
    for (std::size_t i = 0; i < diff.size() && i < state.size() && i < sample.input.size(); ++i) {
        diff[i] = state[i] - sample.input[i];
    }
    const float diff_white = image_white_point(diff);

    const float x0 = left;
    const float x1 = x0 + image_size + gap;
    const float x2 = x1 + image_size + gap;
    const float x3 = x2 + image_size + gap;

    draw_text(renderer, 14, "original", x0, top - 22.0F, SDL_Color{235, 240, 245, 255});
    draw_text(renderer, 14, "input socket", x1, top - 22.0F, SDL_Color{235, 240, 245, 255});
    draw_text(renderer, 14, "VVM state", x2, top - 22.0F, SDL_Color{235, 240, 245, 255});
    draw_text(renderer, 14, "state-original", x3, top - 22.0F, SDL_Color{235, 240, 245, 255});

    draw_state_image(renderer, sample.input, x0, top, panel_scale, original_white, false);
    draw_state_image(renderer, input_view, x1, top, panel_scale, original_white, false);
    draw_state_image(renderer, state, x2, top, panel_scale, state_white, true);
    draw_state_image(renderer, diff, x3, top, panel_scale, diff_white, true);

    std::size_t chosen_rank = 0;
    bool found_rank = false;
    char candidates[256] = "candidates=none";
    if (last_tick != nullptr) {
        char* cursor = candidates;
        std::size_t remaining = sizeof(candidates);
        const int written = std::snprintf(cursor, remaining, "candidates=");
        if (written > 0 && static_cast<std::size_t>(written) < remaining) {
            cursor += written;
            remaining -= static_cast<std::size_t>(written);
        }
        const std::size_t shown = std::min<std::size_t>(last_tick->candidate_indices.size(), 8U);
        for (std::size_t i = 0; i < shown && remaining > 1U; ++i) {
            if (last_tick->candidate_indices[i] == last_tick->chosen_op) {
                chosen_rank = i;
                found_rank = true;
            }
            const int n = std::snprintf(cursor, remaining, "%s%zu", i == 0U ? "" : ",",
                                        last_tick->candidate_indices[i]);
            if (n <= 0 || static_cast<std::size_t>(n) >= remaining) {
                break;
            }
            cursor += n;
            remaining -= static_cast<std::size_t>(n);
        }
    }

    char route[512];
    if (last_tick != nullptr) {
        std::snprintf(
            route, sizeof(route), "op=%zu rank=%s%zu chosen_score=%.4f max_score=%.4f prob=%.4f",
            last_tick->chosen_op, found_rank ? "" : "?", chosen_rank,
            static_cast<double>(last_tick->chosen_score), static_cast<double>(last_tick->max_score),
            static_cast<double>(last_tick->chosen_prob));
    } else {
        std::snprintf(route, sizeof(route), "op=none");
    }

    char line0[256];
    char line1[256];
    char line3[256];
    char line4[256];
    std::snprintf(line0, sizeof(line0), "sample=%zu/%zu  label=%d  tick=%zu  mode=%s  input=%s",
                  sample_index, dataset.test.size(), sample.label, tick_count,
                  paused ? "paused" : "running", input_enabled ? "on" : "off");
    format_clock_detail(line1, sizeof(line1), clock_level);
    std::snprintf(line3, sizeof(line3),
                  "keys: space pause | . step | i input | n/p sample | 0/1 jump | r reset");
    std::snprintf(line4, sizeof(line4),
                  "keys: [/] or -/= clock | q quit    params=%zu ops=%zu d=%zu "
                  "state_white=%.5f",
                  model.parameter_count(), model.config().num_ops, model.config().state_dim,
                  static_cast<double>(state_white));

    const SDL_Color text_color{235, 240, 245, 255};
    draw_text(renderer, 12, line0, 16.0F, 14.0F, text_color);
    draw_text(renderer, 12, line1, 16.0F, 34.0F, text_color);
    draw_text(renderer, 12, route, 16.0F, 54.0F, text_color);
    draw_text(renderer, 12, candidates, 16.0F, 74.0F, text_color);
    draw_text(renderer, 12, line3, 16.0F, 94.0F, text_color);
    draw_text(renderer, 12, line4, 16.0F, 114.0F, text_color);

    char clock_badge[64];
    format_clock_label(clock_badge, sizeof(clock_badge), clock_level);
    const SDL_FRect badge_bg{
        .x = std::max(16.0F, static_cast<float>(width) - 245.0F),
        .y = 14.0F,
        .w = 220.0F,
        .h = 44.0F,
    };
    SDL_SetRenderDrawColor(renderer, 32, 46, 60, 245);
    SDL_RenderFillRect(renderer, &badge_bg);
    draw_text(renderer, 24, clock_badge, badge_bg.x + 12.0F, badge_bg.y + 8.0F,
              SDL_Color{255, 255, 255, 255});
}

std::size_t find_next_label(std::span<const TaskSample> samples, std::size_t start, int label) {
    if (samples.empty()) {
        return 0U;
    }
    for (std::size_t offset = 1; offset <= samples.size(); ++offset) {
        const std::size_t index = (start + offset) % samples.size();
        if (samples[index].label == label) {
            return index;
        }
    }
    return start;
}

std::vector<float> zero_state(std::size_t state_dim) {
    return std::vector<float>(state_dim, 0.0F);
}

struct LiveTrainerState {
    bool running = true;
    bool train_enabled = true;
    bool paused = false;
    bool input_enabled = true;
    bool feedback_enabled = true;
    int train_ticks_per_second = 128;
    int visual_frames_per_second = 60;
    int hard_refractory_ticks = 4;
    float learning_rate = 0.00025F;
    float update_scale = 2.0F;
    float input_scale = 1.0F;
    float activation_threshold = 0.05F;
    float state_heat = 0.0F;
    float op_heat = 0.0F;
    float max_grad_norm = 0.1F;
    float rejection_scale = 0.02F;
    float recency_decay = 1.0F;
    float momentum = 0.9F;
    float display_white = kStateDisplayWhitePoint;
    int default_train_ticks_per_second = 128;
    int default_visual_frames_per_second = 60;
    int default_hard_refractory_ticks = 4;
    float default_learning_rate = 0.00025F;
    float default_update_scale = 2.0F;
    float default_input_scale = 1.0F;
    float default_activation_threshold = 0.05F;
    float default_state_heat = 0.0F;
    float default_op_heat = 0.0F;
    float default_max_grad_norm = 0.1F;
    float default_rejection_scale = 0.02F;
    float default_recency_decay = 1.0F;
    float default_momentum = 0.9F;
    float default_display_white = kStateDisplayWhitePoint;
    std::size_t sample_index = 0;
    std::size_t frame_in_sample = 0;
    std::size_t tick_count = 0;
    float last_loss = 0.0F;
    float last_prediction_error = 0.0F;
    float last_learning_l2 = 0.0F;
    std::size_t last_updated_ops = 0;
    int last_predicted_label = -1;
    float last_prediction_margin = 0.0F;
    float last_class_score0 = 0.0F;
    float last_class_score1 = 0.0F;
    std::size_t rolling_accuracy_window = 128;
    std::size_t rolling_correct = 0;
    std::size_t total_sample_predictions = 0;
    std::size_t total_correct_predictions = 0;
    std::deque<char> rolling_predictions;
    std::array<char, 256> checkpoint_path{
        "artifacts/checkpoints/live_train_hard_refractory.vvmckpt"};
};

void draw_reset_button(const char* id, int& value, int default_value) {
    ImGui::SameLine();
    if (ImGui::SmallButton(id)) {
        value = default_value;
    }
}

void draw_reset_button(const char* id, float& value, float default_value) {
    ImGui::SameLine();
    if (ImGui::SmallButton(id)) {
        value = default_value;
    }
}

struct StateStats {
    float mean = 0.0F;
    float min = 0.0F;
    float max = 0.0F;
    float norm = 0.0F;
};

StateStats state_stats(std::span<const float> values) {
    if (values.empty()) {
        return {};
    }

    StateStats stats{
        .mean = 0.0F,
        .min = std::isfinite(values[0]) ? values[0] : 0.0F,
        .max = std::isfinite(values[0]) ? values[0] : 0.0F,
        .norm = 0.0F,
    };
    float norm_sq = 0.0F;
    for (const float raw_value : values) {
        const float value = std::isfinite(raw_value) ? raw_value : 0.0F;
        stats.mean += value;
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
        norm_sq += value * value;
    }
    stats.mean /= static_cast<float>(values.size());
    stats.norm = std::sqrt(norm_sq);
    return stats;
}

std::vector<float> feedback_input_from_output(std::span<const float> state,
                                              const TaskSample& sample,
                                              const TaskConfig& task_config) {
    constexpr std::size_t kImageDims = 784;
    std::vector<float> input(state.size(), 0.0F);
    if (state.size() < sample.visual_target_offset + kImageDims ||
        state.size() < sample.visual_input_offset + kImageDims) {
        return input;
    }

    for (std::size_t i = 0; i < kImageDims; ++i) {
        const float value = state[sample.visual_target_offset + i];
        input[sample.visual_input_offset + i] = value;
        input[sample.visual_target_offset + i] = task_config.video_input_to_output_scale * value;
    }
    return input;
}

struct LiveClassPrediction {
    int label = -1;
    float best_score = 0.0F;
    float runner_up_score = 0.0F;
};

float live_class_score(std::span<const float> state, const TaskSample& sample, VectorRange range,
                       int candidate) {
    if (sample.class_count <= 0 || sample.class_dims == 0U ||
        sample.class_offset + sample.class_dims > state.size() || candidate < 0 ||
        candidate >= sample.class_count) {
        return 0.0F;
    }

    float score = 0.0F;
    for (std::size_t i = 0; i < sample.class_dims; ++i) {
        const bool matches =
            static_cast<int>(i % static_cast<std::size_t>(sample.class_count)) == candidate;
        const float value = state[sample.class_offset + i];
        if (range == VectorRange::Signed) {
            const float off_value = sample.class_count <= 2
                                        ? -1.0F
                                        : -2.0F / static_cast<float>(sample.class_count - 2);
            score += value * (matches ? 1.0F : off_value);
        } else if (matches) {
            score += value;
        }
    }
    return score;
}

LiveClassPrediction predict_live_class(std::span<const float> state, const TaskSample& sample,
                                       VectorRange range) {
    if (sample.label < 0 || sample.class_count <= 0) {
        return {};
    }

    LiveClassPrediction prediction{
        .label = 0,
        .best_score = live_class_score(state, sample, range, 0),
        .runner_up_score = -std::numeric_limits<float>::infinity(),
    };
    for (int candidate = 1; candidate < sample.class_count; ++candidate) {
        const float score = live_class_score(state, sample, range, candidate);
        if (score > prediction.best_score) {
            prediction.runner_up_score = prediction.best_score;
            prediction.best_score = score;
            prediction.label = candidate;
        } else {
            prediction.runner_up_score = std::max(prediction.runner_up_score, score);
        }
    }
    if (!std::isfinite(prediction.runner_up_score)) {
        prediction.runner_up_score = prediction.best_score;
    }
    return prediction;
}

void record_live_sample_prediction(LiveTrainerState& live, const TaskSample& sample,
                                   std::span<const float> state, const TaskConfig& task_config) {
    const LiveClassPrediction prediction =
        predict_live_class(state, sample, task_config.vector_range);
    live.last_predicted_label = prediction.label;
    live.last_prediction_margin = prediction.best_score - prediction.runner_up_score;
    live.last_class_score0 = live_class_score(state, sample, task_config.vector_range, 0);
    live.last_class_score1 = sample.class_count > 1
                                 ? live_class_score(state, sample, task_config.vector_range, 1)
                                 : 0.0F;
    if (sample.label < 0 || prediction.label < 0) {
        return;
    }

    const char correct = prediction.label == sample.label ? 1 : 0;
    live.rolling_predictions.push_back(correct);
    live.rolling_correct += static_cast<std::size_t>(correct);
    while (live.rolling_predictions.size() > live.rolling_accuracy_window) {
        live.rolling_correct -= static_cast<std::size_t>(live.rolling_predictions.front());
        live.rolling_predictions.pop_front();
    }
    ++live.total_sample_predictions;
    live.total_correct_predictions += static_cast<std::size_t>(correct);
}

void draw_live_scene(SDL_Renderer* renderer, const TaskSample& sample, std::span<const float> state,
                     bool input_enabled, const LiveTrainerState& live, int width, int height) {
    SDL_SetRenderDrawColor(renderer, 11, 13, 18, 255);
    SDL_RenderClear(renderer);

    const float original_white = kMnistDisplayWhitePoint;
    const float state_white = std::max(0.001F, live.display_white);
    const bool temporal_target = sample.label < 0 && !sample.target.empty();
    const std::size_t input_offset = temporal_target ? sample.visual_input_offset : 0U;
    const std::size_t target_offset = temporal_target ? sample.visual_target_offset : 0U;
    const std::span<const float> reference = temporal_target
                                                 ? image_region(sample.target, target_offset)
                                                 : image_region(sample.input, input_offset);
    const std::span<const float> input_region = image_region(sample.input, input_offset);
    const std::span<const float> state_input_region = image_region(state, input_offset);
    const std::span<const float> state_region = image_region(state, target_offset);
    std::vector<float> input_view(sample.input.size(), 0.0F);
    if (input_enabled) {
        input_view = sample.input;
    } else if (live.feedback_enabled && temporal_target) {
        constexpr std::size_t kImageDims = 784;
        for (std::size_t i = 0; i < kImageDims && i < state_region.size() &&
                                sample.visual_input_offset + i < input_view.size();
             ++i) {
            input_view[sample.visual_input_offset + i] = state_region[i];
        }
    }
    const std::span<const float> input_view_region = image_region(input_view, input_offset);
    std::vector<float> diff(784U, 0.0F);
    for (std::size_t i = 0; i < diff.size() && i < state_region.size() && i < reference.size();
         ++i) {
        diff[i] = state_region[i] - reference[i];
    }
    const float diff_white = std::max(0.001F, live.display_white);
    const SDL_Color text_color{235, 240, 245, 255};

    const float top = 100.0F;
    const float gap = 24.0F;
    const float left = 28.0F;
    if (temporal_target) {
        const float panel_scale = std::max(
            3.0F, std::min(static_cast<float>(width) / 178.0F, static_cast<float>(height) / 72.0F));
        const float image_size = 28.0F * panel_scale;
        const float x0 = left;
        const float x1 = x0 + image_size + gap;
        const float x2 = x1 + image_size + gap;
        const float x3 = x2 + image_size + gap;
        const float x4 = x3 + image_size + gap;

        draw_text(renderer, 14, "target next", x0, top - 22.0F, text_color);
        draw_text(renderer, 14, "input socket", x1, top - 22.0F, text_color);
        draw_text(renderer, 14, "output socket", x2, top - 22.0F, text_color);
        draw_text(renderer, 14, "output error", x3, top - 22.0F, text_color);
        draw_text(renderer, 14, "state (in | out)", x4, top - 22.0F, text_color);

        draw_state_image(renderer, reference, x0, top, panel_scale, original_white, false);
        draw_state_image(renderer, input_view_region.empty() ? input_region : input_view_region, x1,
                         top, panel_scale, original_white, false);
        draw_state_image(renderer, state_region, x2, top, panel_scale, state_white, true);
        draw_state_image(renderer, diff, x3, top, panel_scale, diff_white, true);
        draw_state_image(renderer, state_input_region, x4, top, panel_scale, state_white, true);
        draw_state_image(renderer, state_region, x4 + image_size, top, panel_scale, state_white,
                         true);
        SDL_SetRenderDrawColor(renderer, 60, 68, 78, 255);
        SDL_RenderLine(renderer, x4 + image_size, top, x4 + image_size, top + image_size);
        return;
    }

    const float panel_scale = std::max(
        4.0F, std::min(static_cast<float>(width) / 150.0F, static_cast<float>(height) / 62.0F));
    const float image_size = 28.0F * panel_scale;
    const float x0 = left;
    const float x1 = x0 + image_size + gap;
    const float x2 = x1 + image_size + gap;
    const float x3 = x2 + image_size + gap;
    draw_text(renderer, 14, "original", x0, top - 22.0F, text_color);
    draw_text(renderer, 14, "input socket", x1, top - 22.0F, text_color);
    draw_text(renderer, 14, "VVM state", x2, top - 22.0F, text_color);
    draw_text(renderer, 14, "state-original", x3, top - 22.0F, text_color);
    draw_state_image(renderer, reference, x0, top, panel_scale, original_white, false);
    draw_state_image(renderer, input_view_region.empty() ? input_region : input_view_region, x1,
                     top, panel_scale, original_white, false);
    draw_state_image(renderer, state_region, x2, top, panel_scale, state_white, true);
    draw_state_image(renderer, diff, x3, top, panel_scale, diff_white, true);
}

void draw_live_status_imgui(const Model& model, const TaskSample& sample,
                            std::span<const float> state, const Tick* last_tick,
                            const LiveTrainerState& live) {
    ImGui::SetNextWindowPos(ImVec2(470.0F, 360.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(390.0F, 260.0F), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("VVM Live Status")) {
        ImGui::End();
        return;
    }

    const StateStats stats = state_stats(state);
    ImGui::Text("tick %zu", live.tick_count);
    ImGui::Text("sample %zu frame %zu label %d", live.sample_index, live.frame_in_sample,
                sample.label);
    ImGui::Text("run %s train %s input %s feedback %s", live.paused ? "paused" : "running",
                live.train_enabled ? "on" : "off", live.input_enabled ? "on" : "off",
                live.feedback_enabled ? "on" : "off");
    ImGui::Text("ticks/sec %d visual fps %d", live.train_ticks_per_second,
                live.visual_frames_per_second);
    ImGui::Text("params %zu", model.parameter_count());
    ImGui::SeparatorText("State");
    ImGui::Text("norm %.5f mean %.6f", static_cast<double>(stats.norm),
                static_cast<double>(stats.mean));
    ImGui::Text("min %.6f max %.6f white %.5f", static_cast<double>(stats.min),
                static_cast<double>(stats.max), static_cast<double>(live.display_white));
    ImGui::SeparatorText("Learning");
    ImGui::Text("loss %.6f pred %.6f", static_cast<double>(live.last_loss),
                static_cast<double>(live.last_prediction_error));
    ImGui::Text("learn_l2 %.6f updated_ops %zu", static_cast<double>(live.last_learning_l2),
                live.last_updated_ops);
    ImGui::SeparatorText("Sample Prediction");
    const float rolling_accuracy = live.rolling_predictions.empty()
                                       ? 0.0F
                                       : static_cast<float>(live.rolling_correct) /
                                             static_cast<float>(live.rolling_predictions.size());
    const float total_accuracy = live.total_sample_predictions == 0U
                                     ? 0.0F
                                     : static_cast<float>(live.total_correct_predictions) /
                                           static_cast<float>(live.total_sample_predictions);
    ImGui::Text("pred %d label %d margin %.5f", live.last_predicted_label, sample.label,
                static_cast<double>(live.last_prediction_margin));
    ImGui::Text("score0 %.5f score1 %.5f", static_cast<double>(live.last_class_score0),
                static_cast<double>(live.last_class_score1));
    ImGui::Text("rolling %zu/%zu %.1f%%", live.rolling_correct, live.rolling_predictions.size(),
                static_cast<double>(100.0F * rolling_accuracy));
    ImGui::Text("total %zu/%zu %.1f%%", live.total_correct_predictions,
                live.total_sample_predictions, static_cast<double>(100.0F * total_accuracy));
    if (last_tick != nullptr) {
        ImGui::SeparatorText("Route");
        ImGui::Text("op %zu prob %.4f", last_tick->chosen_op,
                    static_cast<double>(last_tick->chosen_prob));
        ImGui::Text("score %.4f max %.4f", static_cast<double>(last_tick->chosen_score),
                    static_cast<double>(last_tick->max_score));
    }
    ImGui::End();
}

void draw_live_imgui(Model& model, LiveTrainerState& live) {
    ImGui::SetNextWindowPos(ImVec2(20.0F, 360.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430.0F, 470.0F), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("VVM Live Trainer")) {
        ImGui::End();
        return;
    }

    ImGui::Text("tick: %zu", live.tick_count);
    ImGui::Checkbox("Paused", &live.paused);
    ImGui::SameLine();
    ImGui::Checkbox("Train", &live.train_enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Input", &live.input_enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Feedback", &live.feedback_enabled);
    if (ImGui::SmallButton("-##ticks_dec")) {
        live.train_ticks_per_second = std::max(1, live.train_ticks_per_second - 1);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("+##ticks_inc")) {
        live.train_ticks_per_second = std::min(1000, live.train_ticks_per_second + 1);
    }
    ImGui::SameLine();
    ImGui::SliderInt("Train ticks / second", &live.train_ticks_per_second, 1, 1000);
    draw_reset_button("D##train_ticks", live.train_ticks_per_second,
                      live.default_train_ticks_per_second);
    ImGui::SliderInt("Visual frames / second", &live.visual_frames_per_second, 1, 144);
    draw_reset_button("D##visual_fps", live.visual_frames_per_second,
                      live.default_visual_frames_per_second);
    ImGui::SliderInt("Hard refractory ticks", &live.hard_refractory_ticks, 0, 64);
    draw_reset_button("D##refractory", live.hard_refractory_ticks,
                      live.default_hard_refractory_ticks);
    ImGui::SeparatorText("Learning");
    ImGui::SliderFloat("Learning rate", &live.learning_rate, 0.0000001F, 0.005F, "%.7f",
                       ImGuiSliderFlags_Logarithmic);
    draw_reset_button("D##lr", live.learning_rate, live.default_learning_rate);
    ImGui::SliderFloat("Momentum", &live.momentum, 0.0F, 0.99F, "%.3f");
    draw_reset_button("D##momentum", live.momentum, live.default_momentum);
    ImGui::SliderFloat("Max grad norm", &live.max_grad_norm, 0.0001F, 2.0F, "%.4f",
                       ImGuiSliderFlags_Logarithmic);
    draw_reset_button("D##grad_norm", live.max_grad_norm, live.default_max_grad_norm);
    ImGui::SliderFloat("Recency decay", &live.recency_decay, 0.0F, 1.0F, "%.3f");
    draw_reset_button("D##recency", live.recency_decay, live.default_recency_decay);
    ImGui::SliderFloat("Rejection scale", &live.rejection_scale, 0.0F, 0.2F, "%.5f");
    draw_reset_button("D##rejection", live.rejection_scale, live.default_rejection_scale);
    ImGui::SeparatorText("Runtime");
    ImGui::SliderFloat("Update scale", &live.update_scale, 0.0F, 4.0F, "%.3f");
    draw_reset_button("D##update", live.update_scale, live.default_update_scale);
    ImGui::SliderFloat("Input scale", &live.input_scale, 0.0F, 4.0F, "%.3f");
    draw_reset_button("D##input_scale", live.input_scale, live.default_input_scale);
    ImGui::SliderFloat("Activation threshold", &live.activation_threshold, 0.0F, 0.1F, "%.4f");
    draw_reset_button("D##activation_threshold", live.activation_threshold,
                      live.default_activation_threshold);
    ImGui::SliderFloat("State heat", &live.state_heat, 0.0F, 0.1F, "%.5f");
    draw_reset_button("D##state_heat", live.state_heat, live.default_state_heat);
    ImGui::SliderFloat("Op heat / selected op", &live.op_heat, 0.0F, 0.002F, "%.7f");
    draw_reset_button("D##op_heat", live.op_heat, live.default_op_heat);
    ImGui::SliderFloat("Display white", &live.display_white, 0.005F, 0.25F, "%.4f",
                       ImGuiSliderFlags_Logarithmic);
    draw_reset_button("D##display_white", live.display_white, live.default_display_white);

    Config& config = model.mutable_config();
    config.update_scale = live.update_scale;
    config.input_scale = live.input_scale;
    config.activation_threshold = live.activation_threshold;
    config.state_heat_stddev = live.state_heat;
    config.hard_refractory_ticks =
        static_cast<std::size_t>(std::max(0, live.hard_refractory_ticks));

    ImGui::SeparatorText("Checkpoint");
    ImGui::InputText("Path", live.checkpoint_path.data(), live.checkpoint_path.size());
    if (ImGui::Button("Save")) {
        config.op_heat_stddev = live.op_heat;
        model.save_checkpoint(live.checkpoint_path.data());
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset runtime")) {
        model.reset_runtime();
    }
    ImGui::Text("loss %.6f  pred %.6f  learn %.6f", static_cast<double>(live.last_loss),
                static_cast<double>(live.last_prediction_error),
                static_cast<double>(live.last_learning_l2));
    ImGui::End();
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

int run_training_visualizer(const Config& config, const TaskConfig& task_config,
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
    const TaskDataset dataset = make_task_dataset(config, task_config);
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
                train_task_epoch(model, dataset.train, dataset.test, task_config, history.size()));
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

int run_probe_visualizer(const Config& config, const TaskConfig& task_config, std::size_t epochs,
                         bool restore_best, bool anchor_to_best, const std::string& load_model_path,
                         const std::string& save_model_path) {
    Model model(config);
    const TaskDataset dataset = make_task_dataset(config, task_config);
    float best_balanced_accuracy = -1.0F;

    if (!load_model_path.empty()) {
        model.load_checkpoint(load_model_path);
        std::cout << "loaded probe model from " << load_model_path << '\n';
    } else {
        const std::vector<float> initial_bank(model.op_bank().begin(), model.op_bank().end());
        std::vector<float> best_bank;
        std::cout << "training probe model for " << epochs << " epochs...\n";
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::span<const float> op_anchor = {};
            if (task_config.op_anchor_scale > 0.0F) {
                if (anchor_to_best && !best_bank.empty()) {
                    op_anchor = best_bank;
                } else if (!anchor_to_best) {
                    op_anchor = initial_bank;
                }
            }
            const LossPoint loss =
                train_task_epoch(model, dataset.train, dataset.test, task_config, epoch, op_anchor);
            std::cout << "epoch " << epoch << " balanced=" << (100.0F * loss.test_balanced_accuracy)
                      << "% class_ce=" << loss.test_class_cross_entropy
                      << " loss=" << loss.test_loss << '\n';
            if (loss.accuracy_samples > 0U &&
                loss.test_balanced_accuracy > best_balanced_accuracy) {
                best_balanced_accuracy = loss.test_balanced_accuracy;
                if (restore_best || anchor_to_best) {
                    best_bank.assign(model.op_bank().begin(), model.op_bank().end());
                }
            }
        }
        if (restore_best && !best_bank.empty()) {
            model.replace_op_bank(best_bank);
        }
        if (!save_model_path.empty()) {
            model.save_checkpoint(save_model_path);
            std::cout << "saved probe model to " << save_model_path << '\n';
        }
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw_sdl_error("SDL_Init failed");
    }

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    if (!SDL_CreateWindowAndRenderer("vvm live probe", 1320, 560, SDL_WINDOW_RESIZABLE, &window,
                                     &renderer)) {
        SDL_Quit();
        throw_sdl_error("SDL_CreateWindowAndRenderer failed");
    }
    init_text_subsystem();

    std::vector<float> state = neutral_state(config.state_dim);
    std::mt19937 rng(config.seed ^ 0xC0FFEEU);
    std::size_t sample_index = 0;
    std::size_t tick_count = 0;
    int clock_level = 0;
    std::size_t render_frame = 0;
    bool input_enabled = true;
    bool paused = false;
    bool running = true;
    bool single_step = false;
    Tick last_tick{};
    bool has_last_tick = false;

    while (running) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                const SDL_Keycode key = event.key.key;
                if (key == SDLK_ESCAPE || key == SDLK_Q) {
                    running = false;
                } else if (key == SDLK_SPACE) {
                    paused = !paused;
                } else if (key == SDLK_I) {
                    input_enabled = !input_enabled;
                } else if (key == SDLK_R) {
                    state = neutral_state(config.state_dim);
                    model.reset_runtime();
                    tick_count = 0;
                    render_frame = 0;
                    has_last_tick = false;
                } else if (key == SDLK_PERIOD) {
                    single_step = true;
                } else if (key == SDLK_N && !dataset.test.empty()) {
                    sample_index = (sample_index + 1U) % dataset.test.size();
                    has_last_tick = false;
                } else if (key == SDLK_P && !dataset.test.empty()) {
                    sample_index =
                        sample_index == 0U ? dataset.test.size() - 1U : sample_index - 1U;
                    has_last_tick = false;
                } else if (key == SDLK_EQUALS || key == SDLK_PLUS || key == SDLK_RIGHTBRACKET) {
                    clock_level = std::min(255, clock_level + 1);
                } else if (key == SDLK_MINUS || key == SDLK_LEFTBRACKET) {
                    clock_level = std::max(-10, clock_level - 1);
                } else if (key == SDLK_0) {
                    sample_index = find_next_label(dataset.test, sample_index, 0);
                    has_last_tick = false;
                } else if (key == SDLK_1) {
                    sample_index = find_next_label(dataset.test, sample_index, 1);
                    has_last_tick = false;
                }
            }
        }

        if (!dataset.test.empty() && (!paused || single_step)) {
            const TaskSample& sample = dataset.test[sample_index % dataset.test.size()];
            std::size_t ticks = 0;
            if (single_step) {
                ticks = 1U;
            } else if (clock_level >= 0) {
                ticks = static_cast<std::size_t>(clock_level + 1);
            } else {
                const std::size_t divisor =
                    static_cast<std::size_t>(1U << std::min(10, -clock_level));
                ticks = render_frame % divisor == 0U ? 1U : 0U;
            }
            for (std::size_t i = 0; i < ticks; ++i) {
                if (input_enabled) {
                    last_tick = model.tick(state, rng, tick_count, sample.input);
                } else {
                    last_tick = model.tick(state, rng, tick_count);
                }
                has_last_tick = true;
                ++tick_count;
            }
            single_step = false;
        }
        ++render_frame;

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        draw_probe(renderer, model, dataset, state, sample_index, input_enabled, paused, tick_count,
                   clock_level, has_last_tick ? &last_tick : nullptr, width, height);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    shutdown_text_subsystem();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

int run_live_training_visualizer(const Config& config, const TaskConfig& task_config) {
    Model model(config);
    const TaskDataset dataset = make_task_dataset(config, task_config);
    if (dataset.train.empty()) {
        throw std::runtime_error("live trainer requires training samples");
    }

    LiveTrainerState live{};
    live.learning_rate = task_config.learning_rate;
    live.momentum = task_config.momentum;
    live.max_grad_norm = task_config.max_grad_norm;
    live.recency_decay = task_config.recency_decay;
    live.rejection_scale = task_config.rejection_scale;
    live.update_scale = config.update_scale;
    live.input_scale = config.input_scale;
    live.activation_threshold = config.activation_threshold;
    live.state_heat = config.state_heat_stddev;
    live.op_heat = config.op_heat_stddev;
    live.hard_refractory_ticks = static_cast<int>(config.hard_refractory_ticks);
    live.default_learning_rate = live.learning_rate;
    live.default_momentum = live.momentum;
    live.default_max_grad_norm = live.max_grad_norm;
    live.default_recency_decay = live.recency_decay;
    live.default_rejection_scale = live.rejection_scale;
    live.default_update_scale = live.update_scale;
    live.default_input_scale = live.input_scale;
    live.default_activation_threshold = live.activation_threshold;
    live.default_state_heat = live.state_heat;
    live.default_op_heat = live.op_heat;
    live.default_hard_refractory_ticks = live.hard_refractory_ticks;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw_sdl_error("SDL_Init failed");
    }

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    constexpr SDL_WindowFlags kWindowFlags =
        static_cast<SDL_WindowFlags>(SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS);
    if (!SDL_CreateWindowAndRenderer("vvm live trainer", 1600, 900, kWindowFlags, &window,
                                     &renderer)) {
        SDL_Quit();
        throw_sdl_error("SDL_CreateWindowAndRenderer failed");
    }
    if (!SDL_SetRenderVSync(renderer, 0)) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        throw_sdl_error("SDL_SetRenderVSync failed");
    }

    init_text_subsystem();
    init_imgui(window, renderer);

    std::vector<float> state = zero_state(model.config().state_dim);
    std::vector<float> display_state = state;
    std::vector<float> visual_state = state;
    std::vector<Tick> window_ticks;
    window_ticks.reserve(task_config.window_size);
    std::mt19937 rng(task_config.seed ^ 0xBADC0DEU);
    Tick last_tick{};
    bool has_last_tick = false;
    bool running = true;
    bool single_step = false;
    std::size_t display_sample_index = 0;
    std::size_t visual_sample_index = 0;

    auto tick_once = [&]() {
        const std::size_t tick_sample_index = live.sample_index % dataset.train.size();
        const TaskSample& sample = dataset.train[tick_sample_index];
        if (live.input_enabled) {
            last_tick = model.tick(state, rng, live.tick_count, sample.input);
        } else if (live.feedback_enabled) {
            const std::vector<float> feedback_input =
                feedback_input_from_output(state, sample, task_config);
            last_tick = model.tick(state, rng, live.tick_count, feedback_input);
        } else {
            last_tick = model.tick(state, rng, live.tick_count);
        }
        display_state = state;
        display_sample_index = tick_sample_index;
        const std::vector<float> target_weights = target_weights_for_frame(
            sample, model.config().state_dim, live.frame_in_sample, task_config);
        apply_observation(last_tick, sample.target, model.config().curiosity_scale, target_weights);
        live.last_prediction_error = last_tick.prediction_error;
        has_last_tick = true;

        if (window_ticks.size() == task_config.window_size) {
            window_ticks.erase(window_ticks.begin());
        }
        window_ticks.push_back(last_tick);

        if (live.train_enabled && !window_ticks.empty()) {
            TrainConfig train_config{};
            train_config.learning_rate = live.learning_rate;
            train_config.momentum = live.momentum;
            train_config.recency_decay = live.recency_decay;
            train_config.max_grad_norm = live.max_grad_norm;
            train_config.rejection_scale = live.rejection_scale;
            train_config.rejection_threshold = task_config.rejection_threshold;
            train_config.rejection_overuse_scale = task_config.rejection_overuse_scale;
            train_config.affinity_retain_scale = task_config.affinity_retain_scale;
            train_config.affinity_retain_threshold = task_config.affinity_retain_threshold;
            train_config.affinity_retain_underuse_scale =
                task_config.affinity_retain_underuse_scale;
            train_config.backprop_through_state = task_config.backprop_through_state;
            const TrainResult result = model.train_window(window_ticks, train_config);
            live.last_loss = result.loss;
            live.last_learning_l2 = result.learning_update_l2;
            live.last_updated_ops = result.updated_ops;
        }

        ++live.tick_count;
        ++live.frame_in_sample;
        if (live.frame_in_sample >= task_config.frames_per_sample) {
            record_live_sample_prediction(live, sample, state, task_config);
            live.sample_index = (live.sample_index + 1U) % dataset.train.size();
            live.frame_in_sample = 0;
        }
    };

    using Clock = std::chrono::steady_clock;
    auto last_clock = Clock::now();
    double tick_accumulator = 0.0;
    double visual_accumulator = 0.0;
    double ui_accumulator = 0.0;
    constexpr double kUiFrameSeconds = 1.0 / 144.0;
    constexpr int kMaxTicksPerLoop = 256;

    while (running) {
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_clock).count();
        last_clock = now;
        const double clamped_elapsed = std::min(elapsed, 0.25);
        if (!live.paused) {
            tick_accumulator += clamped_elapsed;
        }
        visual_accumulator += clamped_elapsed;
        ui_accumulator += clamped_elapsed;

        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN && !ImGui::GetIO().WantCaptureKeyboard) {
                const SDL_Keycode key = event.key.key;
                if (key == SDLK_ESCAPE || key == SDLK_Q) {
                    running = false;
                } else if (key == SDLK_SPACE) {
                    live.paused = !live.paused;
                } else if (key == SDLK_PERIOD) {
                    single_step = true;
                } else if (key == SDLK_I) {
                    live.input_enabled = !live.input_enabled;
                } else if (key == SDLK_F) {
                    live.feedback_enabled = !live.feedback_enabled;
                } else if (key == SDLK_R) {
                    state = zero_state(model.config().state_dim);
                    display_state = state;
                    visual_state = state;
                    model.reset_runtime();
                    window_ticks.clear();
                    live.rolling_predictions.clear();
                    live.rolling_correct = 0;
                    live.total_sample_predictions = 0;
                    live.total_correct_predictions = 0;
                    live.last_predicted_label = -1;
                    live.last_prediction_margin = 0.0F;
                    live.last_class_score0 = 0.0F;
                    live.last_class_score1 = 0.0F;
                    has_last_tick = false;
                } else if (key == SDLK_EQUALS || key == SDLK_PLUS || key == SDLK_RIGHTBRACKET) {
                    live.train_ticks_per_second = std::min(1000, live.train_ticks_per_second * 2);
                } else if (key == SDLK_MINUS || key == SDLK_LEFTBRACKET) {
                    live.train_ticks_per_second = std::max(1, live.train_ticks_per_second / 2);
                }
            }
        }

        Config& mutable_config = model.mutable_config();
        mutable_config.update_scale = live.update_scale;
        mutable_config.state_heat_stddev = live.state_heat;
        mutable_config.op_heat_stddev = live.op_heat;
        mutable_config.hard_refractory_ticks =
            static_cast<std::size_t>(std::max(0, live.hard_refractory_ticks));

        if (single_step) {
            tick_once();
            single_step = false;
            tick_accumulator = 0.0;
        }

        const double tick_seconds =
            1.0 / static_cast<double>(std::max(1, live.train_ticks_per_second));
        int ticks_run = 0;
        while (!live.paused && tick_accumulator >= tick_seconds && ticks_run < kMaxTicksPerLoop) {
            tick_once();
            tick_accumulator -= tick_seconds;
            ++ticks_run;
        }
        if (ticks_run == kMaxTicksPerLoop) {
            tick_accumulator = 0.0;
        }

        const double visual_seconds =
            1.0 / static_cast<double>(std::clamp(live.visual_frames_per_second, 1, 144));
        if (visual_accumulator >= visual_seconds) {
            visual_state = display_state;
            visual_sample_index = display_sample_index;
            visual_accumulator = std::fmod(visual_accumulator, visual_seconds);
        }

        if (ui_accumulator < kUiFrameSeconds) {
            SDL_Delay(1);
            continue;
        }
        ui_accumulator = std::fmod(ui_accumulator, kUiFrameSeconds);

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        const TaskSample& visible_sample =
            dataset.train[visual_sample_index % dataset.train.size()];
        const TaskSample& status_sample =
            dataset.train[display_sample_index % dataset.train.size()];
        draw_live_scene(renderer, visible_sample, visual_state, live.input_enabled, live, width,
                        height);

        new_imgui_frame();
        draw_live_imgui(model, live);
        draw_live_status_imgui(model, status_sample, display_state,
                               has_last_tick ? &last_tick : nullptr, live);
        render_imgui(renderer);
        SDL_RenderPresent(renderer);
    }

    shutdown_imgui();
    shutdown_text_subsystem();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

} // namespace vvm
