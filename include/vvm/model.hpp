#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace vvm {

enum class ActivationKind {
    Relu,
    LeakyRelu,
    Clamp,
    Deadzone,
};

struct Config {
    std::size_t state_dim = 256;
    std::size_t num_ops = 1024;
    std::size_t candidate_count = 8;
    std::size_t sample_candidate_count = 0;
    std::size_t steps = 8;
    ActivationKind activation = ActivationKind::Deadzone;
    float update_scale = 1.0F;
    float input_scale = 1.0F;
    float activation_threshold = 0.05F;
    float activation_leak = 0.01F;
    float retrieval_temperature = 0.0F;
    float state_heat_stddev = 0.0F;
    float op_heat_stddev = 0.0F;
    float heat_decay = 1.0F;
    float curiosity_scale = 1.0F;
    bool sample_retrieval = true;
    std::uint32_t seed = 0xC0FFEEU;
};

struct Retrieval {
    std::vector<std::size_t> candidate_indices;
    std::vector<float> candidate_weights;
    std::vector<float> candidate_scores;
    std::size_t chosen_index = 0;
    float chosen_score = 0.0F;
    float max_score = 0.0F;
};

struct RewardSignal {
    float curiosity_reward = 0.0F;
    float external_reward = 0.0F;
    bool has_external_reward = false;
    float total_reward = 0.0F;
};

struct Tick {
    std::vector<float> state_before;
    std::vector<float> working_pre_state;
    std::vector<float> working_state;
    std::vector<std::size_t> candidate_indices;
    std::vector<float> candidate_probs;
    std::vector<float> candidate_scores;
    std::size_t chosen_op = 0;
    float chosen_prob = 0.0F;
    float chosen_score = 0.0F;
    float max_score = 0.0F;

    std::vector<float> pre_activation;
    std::vector<float> post_activation;
    std::vector<float> predicted_state;
    std::vector<float> observed_state;
    std::vector<float> observation_weights;

    float activation_mean = 0.0F;
    float prediction_error = 0.0F;
    RewardSignal reward;
    float state_heat_stddev = 0.0F;
    float op_heat_stddev = 0.0F;
    float state_heat_l2 = 0.0F;
    float op_heat_l2 = 0.0F;
    std::vector<float> op_heat_l2_by_op;
};

struct StepTrace {
    Retrieval retrieval;
    float state_norm = 0.0F;
    float activation_mean = 0.0F;
    float prediction_error = 0.0F;
    float curiosity_reward = 0.0F;
    float state_heat_stddev = 0.0F;
    float op_heat_stddev = 0.0F;
    float state_heat_l2 = 0.0F;
    float op_heat_l2 = 0.0F;
};

struct RunResult {
    std::vector<float> state;
    std::vector<StepTrace> trace;
};

struct TrainConfig {
    float learning_rate = 0.01F;
    float momentum = 0.0F;
    float recency_decay = 0.97F;
    float max_grad_norm = 1.0F;
    float rejection_scale = 0.0F;
    float rejection_threshold = 0.02F;
    float rejection_overuse_scale = 0.0F;
    float affinity_retain_scale = 0.0F;
    float affinity_retain_threshold = 0.0F;
    float affinity_retain_underuse_scale = 0.0F;
    std::span<const std::size_t> op_usage_counts = {};
    std::span<const float> op_anchor = {};
    float op_anchor_scale = 0.0F;
    bool average_repeated_ops = false;
    bool backprop_through_state = false;
};

struct TrainResult {
    float loss = 0.0F;
    float mean_prediction_error = 0.0F;
    float learning_update_l2 = 0.0F;
    std::vector<float> op_update_l2_by_op;
    std::size_t tick_count = 0;
    std::size_t updated_ops = 0;
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
    void replace_op_bank(std::span<const float> op_bank);
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);

    [[nodiscard]] std::size_t parameter_count() const {
        return op_bank_.size();
    }

    [[nodiscard]] std::size_t parameter_bytes() const {
        return parameter_count() * sizeof(float);
    }

    [[nodiscard]] RunResult run(std::span<const float> initial_state);
    [[nodiscard]] Tick tick(std::vector<float>& state, std::mt19937& rng, std::size_t clock,
                            std::span<const float> input = {},
                            RewardSignal reward = RewardSignal{});
    [[nodiscard]] TrainResult train_window(std::span<const Tick> ticks, TrainConfig train_config);
    [[nodiscard]] std::vector<float> seeded_state(float scale = 1.0F) const;
    [[nodiscard]] std::vector<float> predict_next(std::span<const float> state,
                                                  std::span<const float> input = {}) const;
    [[nodiscard]] static float prediction_error(std::span<const float> predicted,
                                                std::span<const float> observed);
    [[nodiscard]] static float prediction_error(std::span<const float> predicted,
                                                std::span<const float> observed,
                                                std::span<const float> weights);

  private:
    struct Prediction {
        std::vector<float> state;
        std::vector<float> pre_activation;
        std::vector<float> post_activation;
        Retrieval retrieval;
        float activation_mean = 0.0F;
    };

    [[nodiscard]] Retrieval retrieve(std::span<const float> state) const;
    [[nodiscard]] Prediction predict_from_working_state(std::span<const float> working_state) const;
    [[nodiscard]] Prediction predict_from_working_state(std::span<const float> working_state,
                                                        std::mt19937& rng) const;
    [[nodiscard]] std::vector<float> heat_op_bank(float stddev, std::mt19937& rng);
    void normalize_op(std::size_t op);

    Config config_;
    std::vector<float> op_bank_;
    std::vector<float> op_velocity_;
};

[[nodiscard]] float l2_norm(std::span<const float> values);
[[nodiscard]] float dot_product(std::span<const float> a, std::span<const float> b);
void apply_observation(Tick& tick, std::span<const float> observed, float curiosity_scale,
                       std::span<const float> weights = {});

} // namespace vvm
