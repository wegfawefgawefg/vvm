#include "nnvm/model.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nnvm {
namespace {

float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float fast_tanh(float x) {
    return std::tanh(x);
}

float sigmoid(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}

std::vector<float> softmax(std::span<const float> values, float temperature) {
    if (temperature <= 0.0F) {
        throw std::invalid_argument("temperature must be positive");
    }

    const float max_value = *std::max_element(values.begin(), values.end());
    std::vector<float> weights(values.size(), 0.0F);
    float sum = 0.0F;
    for (std::size_t i = 0; i < values.size(); ++i) {
        weights[i] = std::exp((values[i] - max_value) / temperature);
        sum += weights[i];
    }
    for (float& weight : weights) {
        weight /= sum;
    }
    return weights;
}

void layer_norm(std::vector<float>& state) {
    const float mean =
        std::accumulate(state.begin(), state.end(), 0.0F) / static_cast<float>(state.size());

    float variance = 0.0F;
    for (float value : state) {
        const float centered = value - mean;
        variance += centered * centered;
    }
    variance /= static_cast<float>(state.size());

    const float inv_std = 1.0F / std::sqrt(variance + 1.0e-5F);
    for (float& value : state) {
        value = (value - mean) * inv_std;
    }
}

} // namespace

float l2_norm(std::span<const float> values) {
    return std::sqrt(dot(values, values));
}

float cosine_similarity(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("cosine_similarity requires equal vector sizes");
    }

    const float denom = l2_norm(a) * l2_norm(b);
    if (denom <= 1.0e-8F) {
        return 0.0F;
    }
    return dot(a, b) / denom;
}

Model::Model(Config config) : config_(config) {
    if (config_.state_dim == 0U) {
        throw std::invalid_argument("state_dim must be nonzero");
    }
    if (config_.num_ops == 0U) {
        throw std::invalid_argument("num_ops must be nonzero");
    }
    if (config_.top_k == 0U || config_.top_k > config_.num_ops) {
        throw std::invalid_argument("top_k must be in [1, num_ops]");
    }
    if (config_.temperature <= 0.0F) {
        throw std::invalid_argument("temperature must be positive");
    }

    std::mt19937 rng(config_.seed);
    std::normal_distribution<float> init(0.0F, 0.02F);

    op_bank_.resize(config_.num_ops * config_.state_dim);
    gate_weights_.resize(config_.state_dim);
    delta_weights_.resize(config_.state_dim);

    for (float& value : op_bank_) {
        value = init(rng);
    }
    for (float& value : gate_weights_) {
        value = init(rng);
    }
    for (float& value : delta_weights_) {
        value = init(rng);
    }
}

std::vector<float> Model::seeded_state(float scale) const {
    std::mt19937 rng(config_.seed ^ 0x9E3779B9U);
    std::normal_distribution<float> init(0.0F, scale);

    std::vector<float> state(config_.state_dim);
    for (float& value : state) {
        value = init(rng);
    }
    layer_norm(state);
    return state;
}

Retrieval Model::retrieve(std::span<const float> state) const {
    if (state.size() != config_.state_dim) {
        throw std::invalid_argument("state size does not match model state_dim");
    }

    std::vector<std::pair<float, std::size_t>> scores;
    scores.reserve(config_.num_ops);

    for (std::size_t op = 0; op < config_.num_ops; ++op) {
        const std::span<const float> op_vector(op_bank_.data() + (op * config_.state_dim),
                                               config_.state_dim);
        scores.emplace_back(cosine_similarity(state, op_vector), op);
    }

    std::partial_sort(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(config_.top_k),
                      scores.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    std::vector<float> top_scores(config_.top_k);
    Retrieval retrieval{};
    retrieval.indices.resize(config_.top_k);
    retrieval.max_score = scores.front().first;

    for (std::size_t i = 0; i < config_.top_k; ++i) {
        top_scores[i] = scores[i].first;
        retrieval.indices[i] = scores[i].second;
    }
    retrieval.weights = softmax(top_scores, config_.temperature);
    return retrieval;
}

StepTrace Model::step(std::vector<float>& state) const {
    Retrieval retrieval = retrieve(state);

    std::vector<float> mixed_op(config_.state_dim, 0.0F);
    for (std::size_t rank = 0; rank < retrieval.indices.size(); ++rank) {
        const std::size_t op = retrieval.indices[rank];
        const float weight = retrieval.weights[rank];
        const std::span<const float> op_vector(op_bank_.data() + (op * config_.state_dim),
                                               config_.state_dim);

        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            mixed_op[i] += weight * op_vector[i];
        }
    }

    float gate_sum = 0.0F;
    for (std::size_t i = 0; i < config_.state_dim; ++i) {
        const float combined = state[i] + mixed_op[i];
        const float gate = sigmoid(gate_weights_[i] * combined);
        const float delta = fast_tanh(delta_weights_[i] * (state[i] - mixed_op[i]));
        state[i] += gate * delta;
        gate_sum += gate;
    }

    layer_norm(state);

    return StepTrace{
        .retrieval = std::move(retrieval),
        .state_norm = l2_norm(state),
        .gate_mean = gate_sum / static_cast<float>(config_.state_dim),
    };
}

RunResult Model::run(std::span<const float> initial_state) const {
    std::vector<float> state(initial_state.begin(), initial_state.end());
    if (state.size() != config_.state_dim) {
        throw std::invalid_argument("initial state size does not match model state_dim");
    }

    RunResult result{};
    result.trace.reserve(config_.steps);

    for (std::size_t step_index = 0; step_index < config_.steps; ++step_index) {
        result.trace.push_back(step(state));
    }

    result.state = std::move(state);
    return result;
}

} // namespace nnvm
