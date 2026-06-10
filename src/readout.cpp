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

MlpReadout::MlpReadout(ReadoutConfig config) : config_(config) {
    if (config_.input_dim == 0U) {
        throw std::invalid_argument("mlp readout input_dim must be nonzero");
    }
    if (config_.hidden_dim == 0U) {
        throw std::invalid_argument("mlp readout hidden_dim must be nonzero");
    }
    if (config_.class_count == 0U) {
        throw std::invalid_argument("mlp readout class_count must be nonzero");
    }
    if (config_.learning_rate < 0.0F) {
        throw std::invalid_argument("mlp readout learning_rate must be nonnegative");
    }

    std::mt19937 rng(config_.seed);
    const float input_scale = std::sqrt(2.0F / static_cast<float>(config_.input_dim));
    const float output_scale = std::sqrt(2.0F / static_cast<float>(config_.hidden_dim));
    std::normal_distribution<float> input_init(0.0F, input_scale);
    std::normal_distribution<float> output_init(0.0F, output_scale);

    input_hidden_weights_.resize(config_.input_dim * config_.hidden_dim);
    hidden_biases_.assign(config_.hidden_dim, 0.0F);
    hidden_output_weights_.resize(config_.hidden_dim * config_.class_count);
    output_biases_.assign(config_.class_count, 0.0F);

    for (float& value : input_hidden_weights_) {
        value = input_init(rng);
    }
    for (float& value : hidden_output_weights_) {
        value = output_init(rng);
    }
}

std::vector<float> MlpReadout::logits(std::span<const float> input) const {
    if (input.size() != config_.input_dim) {
        throw std::invalid_argument("mlp readout input size mismatch");
    }

    std::vector<float> hidden(config_.hidden_dim, 0.0F);
    for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
        float value = hidden_biases_[h];
        const std::size_t offset = h * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            value += input_hidden_weights_[offset + i] * input[i];
        }
        hidden[h] = std::max(0.0F, value);
    }

    std::vector<float> out(config_.class_count, 0.0F);
    for (std::size_t klass = 0; klass < config_.class_count; ++klass) {
        float value = output_biases_[klass];
        const std::size_t offset = klass * config_.hidden_dim;
        for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
            value += hidden_output_weights_[offset + h] * hidden[h];
        }
        out[klass] = value;
    }
    return out;
}

int MlpReadout::predict(std::span<const float> input) const {
    const std::vector<float> values = logits(input);
    return static_cast<int>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

float MlpReadout::loss_one(std::span<const float> input, int label) const {
    if (label < 0 || static_cast<std::size_t>(label) >= config_.class_count) {
        throw std::invalid_argument("mlp readout label out of range");
    }

    std::vector<float> values = logits(input);
    const float max_logit = *std::max_element(values.begin(), values.end());
    float sum_exp = 0.0F;
    for (float& value : values) {
        value = std::exp(value - max_logit);
        sum_exp += value;
    }
    if (sum_exp <= 0.0F) {
        throw std::runtime_error("mlp readout softmax underflow");
    }

    const float probability = std::max(values[static_cast<std::size_t>(label)] / sum_exp, 1.0e-8F);
    return -std::log(probability);
}

float MlpReadout::train_one(std::span<const float> input, int label) {
    if (label < 0 || static_cast<std::size_t>(label) >= config_.class_count) {
        throw std::invalid_argument("mlp readout label out of range");
    }
    if (input.size() != config_.input_dim) {
        throw std::invalid_argument("mlp readout input size mismatch");
    }

    std::vector<float> hidden_pre(config_.hidden_dim, 0.0F);
    std::vector<float> hidden(config_.hidden_dim, 0.0F);
    for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
        float value = hidden_biases_[h];
        const std::size_t offset = h * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            value += input_hidden_weights_[offset + i] * input[i];
        }
        hidden_pre[h] = value;
        hidden[h] = std::max(0.0F, value);
    }

    std::vector<float> values(config_.class_count, 0.0F);
    for (std::size_t klass = 0; klass < config_.class_count; ++klass) {
        float value = output_biases_[klass];
        const std::size_t offset = klass * config_.hidden_dim;
        for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
            value += hidden_output_weights_[offset + h] * hidden[h];
        }
        values[klass] = value;
    }

    const float max_logit = *std::max_element(values.begin(), values.end());
    float sum_exp = 0.0F;
    for (float& value : values) {
        value = std::exp(value - max_logit);
        sum_exp += value;
    }
    if (sum_exp <= 0.0F) {
        throw std::runtime_error("mlp readout softmax underflow");
    }
    for (float& value : values) {
        value /= sum_exp;
    }

    const float probability = std::max(values[static_cast<std::size_t>(label)], 1.0e-8F);
    const float loss = -std::log(probability);

    std::vector<float> grad_hidden(config_.hidden_dim, 0.0F);
    std::vector<float> old_hidden_output_weights = hidden_output_weights_;

    for (std::size_t klass = 0; klass < config_.class_count; ++klass) {
        const float target = klass == static_cast<std::size_t>(label) ? 1.0F : 0.0F;
        const float grad = values[klass] - target;
        const std::size_t offset = klass * config_.hidden_dim;
        for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
            grad_hidden[h] += old_hidden_output_weights[offset + h] * grad;
            hidden_output_weights_[offset + h] -= config_.learning_rate * grad * hidden[h];
        }
        output_biases_[klass] -= config_.learning_rate * grad;
    }

    for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
        if (hidden_pre[h] <= 0.0F) {
            continue;
        }
        const float grad = grad_hidden[h];
        const std::size_t offset = h * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            input_hidden_weights_[offset + i] -= config_.learning_rate * grad * input[i];
        }
        hidden_biases_[h] -= config_.learning_rate * grad;
    }

    return loss;
}

std::size_t MlpReadout::parameter_count() const {
    return input_hidden_weights_.size() + hidden_biases_.size() + hidden_output_weights_.size() +
           output_biases_.size();
}

MlpAutoencoder::MlpAutoencoder(ReadoutConfig config) : config_(config) {
    if (config_.input_dim == 0U) {
        throw std::invalid_argument("mlp autoencoder input_dim must be nonzero");
    }
    if (config_.hidden_dim == 0U) {
        throw std::invalid_argument("mlp autoencoder hidden_dim must be nonzero");
    }
    if (config_.learning_rate < 0.0F) {
        throw std::invalid_argument("mlp autoencoder learning_rate must be nonnegative");
    }

    std::mt19937 rng(config_.seed);
    const float input_scale = std::sqrt(2.0F / static_cast<float>(config_.input_dim));
    const float output_scale = std::sqrt(2.0F / static_cast<float>(config_.hidden_dim));
    std::normal_distribution<float> input_init(0.0F, input_scale);
    std::normal_distribution<float> output_init(0.0F, output_scale);

    input_hidden_weights_.resize(config_.input_dim * config_.hidden_dim);
    hidden_biases_.assign(config_.hidden_dim, 0.0F);
    hidden_output_weights_.resize(config_.hidden_dim * config_.input_dim);
    output_biases_.assign(config_.input_dim, 0.0F);

    for (float& value : input_hidden_weights_) {
        value = input_init(rng);
    }
    for (float& value : hidden_output_weights_) {
        value = output_init(rng);
    }
}

std::vector<float> MlpAutoencoder::predict(std::span<const float> input) const {
    if (input.size() != config_.input_dim) {
        throw std::invalid_argument("mlp autoencoder input size mismatch");
    }

    std::vector<float> hidden(config_.hidden_dim, 0.0F);
    for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
        float value = hidden_biases_[h];
        const std::size_t offset = h * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            value += input_hidden_weights_[offset + i] * input[i];
        }
        hidden[h] = std::max(0.0F, value);
    }

    std::vector<float> out(config_.input_dim, 0.0F);
    for (std::size_t i = 0; i < config_.input_dim; ++i) {
        float value = output_biases_[i];
        const std::size_t offset = i * config_.hidden_dim;
        for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
            value += hidden_output_weights_[offset + h] * hidden[h];
        }
        out[i] = value;
    }
    return out;
}

float MlpAutoencoder::loss_one(std::span<const float> input, std::span<const float> target) const {
    if (target.size() != config_.input_dim) {
        throw std::invalid_argument("mlp autoencoder target size mismatch");
    }

    const std::vector<float> out = predict(input);
    float loss = 0.0F;
    for (std::size_t i = 0; i < config_.input_dim; ++i) {
        const float error = out[i] - target[i];
        loss += error * error;
    }
    return loss / static_cast<float>(config_.input_dim);
}

float MlpAutoencoder::train_one(std::span<const float> input, std::span<const float> target) {
    if (input.size() != config_.input_dim || target.size() != config_.input_dim) {
        throw std::invalid_argument("mlp autoencoder vector size mismatch");
    }

    std::vector<float> hidden_pre(config_.hidden_dim, 0.0F);
    std::vector<float> hidden(config_.hidden_dim, 0.0F);
    for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
        float value = hidden_biases_[h];
        const std::size_t offset = h * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            value += input_hidden_weights_[offset + i] * input[i];
        }
        hidden_pre[h] = value;
        hidden[h] = std::max(0.0F, value);
    }

    std::vector<float> out(config_.input_dim, 0.0F);
    for (std::size_t i = 0; i < config_.input_dim; ++i) {
        float value = output_biases_[i];
        const std::size_t offset = i * config_.hidden_dim;
        for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
            value += hidden_output_weights_[offset + h] * hidden[h];
        }
        out[i] = value;
    }

    float loss = 0.0F;
    std::vector<float> grad_hidden(config_.hidden_dim, 0.0F);
    const std::vector<float> old_hidden_output_weights = hidden_output_weights_;
    const float inv_dim = 1.0F / static_cast<float>(config_.input_dim);

    for (std::size_t i = 0; i < config_.input_dim; ++i) {
        const float error = out[i] - target[i];
        loss += error * error;
        const float grad = 2.0F * error * inv_dim;
        const std::size_t offset = i * config_.hidden_dim;
        for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
            grad_hidden[h] += old_hidden_output_weights[offset + h] * grad;
            hidden_output_weights_[offset + h] -= config_.learning_rate * grad * hidden[h];
        }
        output_biases_[i] -= config_.learning_rate * grad;
    }

    for (std::size_t h = 0; h < config_.hidden_dim; ++h) {
        if (hidden_pre[h] <= 0.0F) {
            continue;
        }
        const float grad = grad_hidden[h];
        const std::size_t offset = h * config_.input_dim;
        for (std::size_t i = 0; i < config_.input_dim; ++i) {
            input_hidden_weights_[offset + i] -= config_.learning_rate * grad * input[i];
        }
        hidden_biases_[h] -= config_.learning_rate * grad;
    }

    return loss * inv_dim;
}

std::size_t MlpAutoencoder::parameter_count() const {
    return input_hidden_weights_.size() + hidden_biases_.size() + hidden_output_weights_.size() +
           output_biases_.size();
}

} // namespace vvm
