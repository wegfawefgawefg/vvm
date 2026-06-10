#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vvm {

struct Config {
    std::size_t state_dim = 256;
    std::size_t num_ops = 1024;
    std::size_t top_k = 8;
    std::size_t steps = 8;
    float temperature = 1.0F;
    std::uint32_t seed = 0xC0FFEEU;
};

struct Retrieval {
    std::vector<std::size_t> indices;
    std::vector<float> weights;
    float max_score = 0.0F;
};

struct StepTrace {
    Retrieval retrieval;
    float state_norm = 0.0F;
    float gate_mean = 0.0F;
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

    [[nodiscard]] RunResult run(std::span<const float> initial_state) const;
    [[nodiscard]] std::vector<float> seeded_state(float scale = 1.0F) const;

  private:
    [[nodiscard]] Retrieval retrieve(std::span<const float> state) const;
    [[nodiscard]] StepTrace step(std::vector<float>& state) const;

    Config config_;
    std::vector<float> op_bank_;
    std::vector<float> gate_weights_;
    std::vector<float> delta_weights_;
};

[[nodiscard]] float l2_norm(std::span<const float> values);
[[nodiscard]] float cosine_similarity(std::span<const float> a, std::span<const float> b);

} // namespace vvm
