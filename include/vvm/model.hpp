#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace vvm {

struct Config {
    std::size_t state_dim = 256;
    std::size_t num_ops = 1024;
    std::size_t candidate_count = 8;
    std::size_t steps = 8;
    float update_scale = 1.0F;
    float input_scale = 1.0F;
    float state_heat_stddev = 0.0F;
    float op_heat_stddev = 0.0F;
    float heat_decay = 1.0F;
    float curiosity_scale = 1.0F;
    std::uint32_t seed = 0xC0FFEEU;
};

struct Retrieval {
    std::vector<std::size_t> candidate_indices;
    std::vector<float> candidate_weights;
    std::size_t chosen_index = 0;
    float chosen_score = 0.0F;
    float max_score = 0.0F;
};

struct StepTrace {
    Retrieval retrieval;
    float state_norm = 0.0F;
    float activation_mean = 0.0F;
    float prediction_error = 0.0F;
    float curiosity_reward = 0.0F;
    float state_heat_stddev = 0.0F;
    float op_heat_stddev = 0.0F;
};

struct RunResult {
    std::vector<float> state;
    std::vector<StepTrace> trace;
};

class Model {
  public:
    explicit Model(Config config);

    [[nodiscard]] const Config& config() const {
        return config_;
    }

    [[nodiscard]] std::span<const float> op_bank() const {
        return op_bank_;
    }

    [[nodiscard]] RunResult run(std::span<const float> initial_state);
    [[nodiscard]] std::vector<float> seeded_state(float scale = 1.0F) const;
    [[nodiscard]] std::vector<float> predict_next(std::span<const float> state,
                                                  std::span<const float> input = {}) const;

    [[nodiscard]] static float prediction_error(std::span<const float> predicted,
                                                std::span<const float> observed);

  private:
    struct Prediction {
        std::vector<float> state;
        Retrieval retrieval;
        float activation_mean = 0.0F;
    };

    [[nodiscard]] Retrieval retrieve(std::span<const float> state) const;
    [[nodiscard]] Prediction predict_from_working_state(std::span<const float> working_state) const;
    [[nodiscard]] Prediction predict_from_working_state(std::span<const float> working_state,
                                                        std::mt19937& rng) const;
    [[nodiscard]] StepTrace step(std::vector<float>& state, std::mt19937& rng, std::size_t clock,
                                 std::span<const float> input = {});
    void heat_op_bank(float stddev, std::mt19937& rng);

    Config config_;
    std::vector<float> op_bank_;
};

[[nodiscard]] float l2_norm(std::span<const float> values);
[[nodiscard]] float dot_product(std::span<const float> a, std::span<const float> b);

} // namespace vvm
