#pragma once

#include "vvm/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vvm {

struct ToySample {
    std::vector<float> input;
    std::vector<float> target;
};

struct ToyDataset {
    std::vector<ToySample> train;
    std::vector<ToySample> test;
};

struct ToyTaskConfig {
    std::size_t train_samples = 64;
    std::size_t test_samples = 32;
    std::size_t frames_per_sample = 8;
    std::size_t idle_frames_between_samples = 0;
    std::size_t window_size = 8;
    float learning_rate = 0.05F;
    float recency_decay = 0.97F;
    float max_grad_norm = 1.0F;
    std::uint32_t seed = 0x51A7E5U;
};

struct LossPoint {
    float train_loss = 0.0F;
    float self_loss = 0.0F;
    float test_loss = 0.0F;
};

[[nodiscard]] ToyDataset make_toy_dataset(const Config& model_config,
                                          const ToyTaskConfig& task_config);
[[nodiscard]] float evaluate_toy_loss(Model& model, std::span<const ToySample> samples,
                                      const ToyTaskConfig& task_config);
[[nodiscard]] LossPoint train_toy_epoch(Model& model, std::span<const ToySample> train_samples,
                                        std::span<const ToySample> test_samples,
                                        const ToyTaskConfig& task_config, std::size_t epoch);
[[nodiscard]] std::vector<float> neutral_state(std::size_t state_dim);

} // namespace vvm
