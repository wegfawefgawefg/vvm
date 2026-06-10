#include "vvm/readout.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <stdexcept>

namespace vvm {

LinearReadout::LinearReadout(ReadoutConfig config) : config_(config) {
    if (config_.input_dim == 0U) {
        throw std::invalid_argument("readout input_dim must be nonzero");
    }
    if (config_.class_count == 0U) {
        throw std::invalid_argument("readout class_count must be nonzero");
    }
    if (config_.learning_rate < 0.0F) {
        throw std::invalid_argument("readout learning_rate must be nonnegative");
    }

    std::mt19937 rng(config_.seed);
    std::normal_distribution<float> init(0.0F, 0.01F);
    weights_.resize(config_.input_dim * config_.class_count);
    biases_.assign(config_.class_count, 0.0F);
    for (float& value : weights_) {
        value = init(rng);
    }
}

std::vector<float> LinearReadout::logits(std::span<const float> input) const {
    if (input.size() != config_.input_dim) {
        throw std::invalid_argument("readout input size mismatch");
    }

    std::vector<float> out(config_.class_count, 0.0F);
    for (std::size_t klass = 0; klass < config_.class_count; ++klass) {
        float value = biases_[klass];
        const std::size_t offset = klass * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            value += weights_[offset + i] * input[i];
        }
        out[klass] = value;
    }
    return out;
}

int LinearReadout::predict(std::span<const float> input) const {
    const std::vector<float> values = logits(input);
    return static_cast<int>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

float LinearReadout::loss_one(std::span<const float> input, int label) const {
    if (label < 0 || static_cast<std::size_t>(label) >= config_.class_count) {
        throw std::invalid_argument("readout label out of range");
    }

    std::vector<float> values = logits(input);
    const float max_logit = *std::max_element(values.begin(), values.end());
    float sum_exp = 0.0F;
    for (float& value : values) {
        value = std::exp(value - max_logit);
        sum_exp += value;
    }

    if (sum_exp <= 0.0F) {
        throw std::runtime_error("readout softmax underflow");
    }

    const float probability = std::max(values[static_cast<std::size_t>(label)] / sum_exp, 1.0e-8F);
    return -std::log(probability);
}

float LinearReadout::train_one(std::span<const float> input, int label) {
    if (label < 0 || static_cast<std::size_t>(label) >= config_.class_count) {
        throw std::invalid_argument("readout label out of range");
    }

    std::vector<float> values = logits(input);
    const float max_logit = *std::max_element(values.begin(), values.end());
    float sum_exp = 0.0F;
    for (float& value : values) {
        value = std::exp(value - max_logit);
        sum_exp += value;
    }

    if (sum_exp <= 0.0F) {
        throw std::runtime_error("readout softmax underflow");
    }

    for (float& value : values) {
        value /= sum_exp;
    }

    const float probability = std::max(values[static_cast<std::size_t>(label)], 1.0e-8F);
    const float loss = -std::log(probability);

    for (std::size_t klass = 0; klass < config_.class_count; ++klass) {
        const float target = klass == static_cast<std::size_t>(label) ? 1.0F : 0.0F;
        const float grad = values[klass] - target;
        const std::size_t offset = klass * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            weights_[offset + i] -= config_.learning_rate * grad * input[i];
        }
        biases_[klass] -= config_.learning_rate * grad;
    }

    return loss;
}

std::size_t LinearReadout::parameter_count() const {
    return weights_.size() + biases_.size();
}

} // namespace vvm
