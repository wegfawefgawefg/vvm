#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vvm {

struct ReadoutConfig {
    std::size_t input_dim = 0;
    std::size_t hidden_dim = 64;
    std::size_t class_count = 0;
    float learning_rate = 0.01F;
    std::uint32_t seed = 0xD16A7E5U;
};

struct ReadoutMetrics {
    float loss = 0.0F;
    float accuracy = 0.0F;
    std::size_t samples = 0;
};

class LinearReadout {
  public:
    explicit LinearReadout(ReadoutConfig config);

    [[nodiscard]] const ReadoutConfig& config() const {
        return config_;
    }

    [[nodiscard]] std::vector<float> logits(std::span<const float> input) const;
    [[nodiscard]] float loss_one(std::span<const float> input, int label) const;
    [[nodiscard]] int predict(std::span<const float> input) const;
    [[nodiscard]] float train_one(std::span<const float> input, int label);
    [[nodiscard]] std::size_t parameter_count() const;

  private:
    ReadoutConfig config_;
    std::vector<float> weights_;
    std::vector<float> biases_;
};

class MlpReadout {
  public:
    explicit MlpReadout(ReadoutConfig config);

    [[nodiscard]] const ReadoutConfig& config() const {
        return config_;
    }

    [[nodiscard]] std::vector<float> logits(std::span<const float> input) const;
    [[nodiscard]] float loss_one(std::span<const float> input, int label) const;
    [[nodiscard]] int predict(std::span<const float> input) const;
    [[nodiscard]] float train_one(std::span<const float> input, int label);
    [[nodiscard]] std::size_t parameter_count() const;

  private:
    ReadoutConfig config_;
    std::vector<float> input_hidden_weights_;
    std::vector<float> hidden_biases_;
    std::vector<float> hidden_output_weights_;
    std::vector<float> output_biases_;
};

class MlpAutoencoder {
  public:
    explicit MlpAutoencoder(ReadoutConfig config);

    [[nodiscard]] const ReadoutConfig& config() const {
        return config_;
    }

    [[nodiscard]] std::vector<float> predict(std::span<const float> input) const;
    [[nodiscard]] float loss_one(std::span<const float> input, std::span<const float> target) const;
    [[nodiscard]] float train_one(std::span<const float> input, std::span<const float> target);
    [[nodiscard]] std::size_t parameter_count() const;

  private:
    ReadoutConfig config_;
    std::vector<float> input_hidden_weights_;
    std::vector<float> hidden_biases_;
    std::vector<float> hidden_output_weights_;
    std::vector<float> output_biases_;
};

} // namespace vvm
