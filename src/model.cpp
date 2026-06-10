#include "vvm/model.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace vvm {
namespace {

float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void normalize_l2(std::span<float> state) {
    const float norm = l2_norm(state);
    if (norm <= 1.0e-8F) {
        const float fill = 1.0F / std::sqrt(static_cast<float>(state.size()));
        std::fill(state.begin(), state.end(), fill);
        return;
    }

    const float inv_std = 1.0F / norm;
    for (float& value : state) {
        value *= inv_std;
    }
}

float decayed(float value, float decay, std::size_t clock) {
    if (value <= 0.0F) {
        return 0.0F;
    }
    return value * std::pow(decay, static_cast<float>(clock));
}

void apply_heat(std::span<float> values, float stddev, std::mt19937& rng) {
    if (stddev <= 0.0F) {
        return;
    }

    std::normal_distribution<float> noise(0.0F, stddev);
    for (float& value : values) {
        value += noise(rng);
    }
}

} // namespace

float l2_norm(std::span<const float> values) {
    return std::sqrt(dot(values, values));
}

float dot_product(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot_product requires equal vector sizes");
    }
    return dot(a, b);
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
    if (config_.update_scale < 0.0F) {
        throw std::invalid_argument("update_scale must be nonnegative");
    }
    if (config_.input_scale < 0.0F) {
        throw std::invalid_argument("input_scale must be nonnegative");
    }
    if (config_.state_heat_stddev < 0.0F) {
        throw std::invalid_argument("state_heat_stddev must be nonnegative");
    }
    if (config_.op_heat_stddev < 0.0F) {
        throw std::invalid_argument("op_heat_stddev must be nonnegative");
    }
    if (config_.heat_decay < 0.0F || config_.heat_decay > 1.0F) {
        throw std::invalid_argument("heat_decay must be in [0, 1]");
    }
    if (config_.curiosity_scale < 0.0F) {
        throw std::invalid_argument("curiosity_scale must be nonnegative");
    }

    std::mt19937 rng(config_.seed);
    std::normal_distribution<float> init(0.0F, 0.02F);

    op_bank_.resize(config_.num_ops * config_.state_dim);

    for (float& value : op_bank_) {
        value = init(rng);
    }

    for (std::size_t op = 0; op < config_.num_ops; ++op) {
        std::span<float> op_vector(op_bank_.data() + (op * config_.state_dim), config_.state_dim);
        normalize_l2(op_vector);
    }
}

std::vector<float> Model::seeded_state(float scale) const {
    std::mt19937 rng(config_.seed ^ 0x9E3779B9U);
    std::normal_distribution<float> init(0.0F, scale);

    std::vector<float> state(config_.state_dim);
    for (float& value : state) {
        value = init(rng);
    }
    normalize_l2(state);
    return state;
}

std::vector<float> Model::predict_next(std::span<const float> state,
                                       std::span<const float> input) const {
    if (state.size() != config_.state_dim) {
        throw std::invalid_argument("state size does not match model state_dim");
    }
    if (!input.empty() && input.size() != config_.state_dim) {
        throw std::invalid_argument("input size must match model state_dim");
    }

    std::vector<float> working_state(state.begin(), state.end());
    for (std::size_t i = 0; i < input.size(); ++i) {
        working_state[i] += config_.input_scale * input[i];
    }
    normalize_l2(working_state);

    return predict_from_working_state(working_state).state;
}

float Model::prediction_error(std::span<const float> predicted, std::span<const float> observed) {
    if (predicted.size() != observed.size()) {
        throw std::invalid_argument("prediction_error requires equal vector sizes");
    }

    float sum = 0.0F;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        const float error = observed[i] - predicted[i];
        sum += error * error;
    }
    return sum / static_cast<float>(predicted.size());
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
        scores.emplace_back(dot_product(state, op_vector), op);
    }

    std::partial_sort(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(config_.top_k),
                      scores.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    Retrieval retrieval{};
    retrieval.indices.resize(config_.top_k);
    retrieval.weights.resize(config_.top_k, 1.0F / static_cast<float>(config_.top_k));
    retrieval.max_score = scores.front().first;

    for (std::size_t i = 0; i < config_.top_k; ++i) {
        retrieval.indices[i] = scores[i].second;
    }
    return retrieval;
}

Model::Prediction Model::predict_from_working_state(std::span<const float> working_state) const {
    Retrieval retrieval = retrieve(working_state);
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

    std::vector<float> predicted_state(working_state.begin(), working_state.end());
    float activation_sum = 0.0F;
    for (std::size_t i = 0; i < config_.state_dim; ++i) {
        predicted_state[i] =
            std::max(0.0F, predicted_state[i] + (config_.update_scale * mixed_op[i]));
        activation_sum += predicted_state[i];
    }
    normalize_l2(predicted_state);

    return Prediction{
        .state = std::move(predicted_state),
        .retrieval = std::move(retrieval),
        .activation_mean = activation_sum / static_cast<float>(config_.state_dim),
    };
}

StepTrace Model::step(std::vector<float>& state, std::mt19937& rng, std::size_t clock,
                      std::span<const float> input) {
    if (!input.empty() && input.size() != config_.state_dim) {
        throw std::invalid_argument("input size must match model state_dim");
    }

    const float op_heat = decayed(config_.op_heat_stddev, config_.heat_decay, clock);
    heat_op_bank(op_heat, rng);

    std::vector<float> working_state(state.begin(), state.end());
    for (std::size_t i = 0; i < input.size(); ++i) {
        working_state[i] += config_.input_scale * input[i];
    }
    normalize_l2(working_state);

    Prediction prediction = predict_from_working_state(working_state);
    state = prediction.state;

    const float state_heat = decayed(config_.state_heat_stddev, config_.heat_decay, clock);
    apply_heat(state, state_heat, rng);
    normalize_l2(state);

    const float error = prediction_error(prediction.state, state);

    return StepTrace{
        .retrieval = std::move(prediction.retrieval),
        .state_norm = l2_norm(state),
        .activation_mean = prediction.activation_mean,
        .prediction_error = error,
        .curiosity_reward = config_.curiosity_scale * error,
        .state_heat_stddev = state_heat,
        .op_heat_stddev = op_heat,
    };
}

void Model::heat_op_bank(float stddev, std::mt19937& rng) {
    if (stddev <= 0.0F) {
        return;
    }

    apply_heat(op_bank_, stddev, rng);
    for (std::size_t op = 0; op < config_.num_ops; ++op) {
        std::span<float> op_vector(op_bank_.data() + (op * config_.state_dim), config_.state_dim);
        normalize_l2(op_vector);
    }
}

RunResult Model::run(std::span<const float> initial_state) {
    std::vector<float> state(initial_state.begin(), initial_state.end());
    if (state.size() != config_.state_dim) {
        throw std::invalid_argument("initial state size does not match model state_dim");
    }

    std::mt19937 rng(config_.seed ^ 0xA341316CU);

    RunResult result{};
    result.trace.reserve(config_.steps);

    for (std::size_t step_index = 0; step_index < config_.steps; ++step_index) {
        result.trace.push_back(step(state, rng, step_index));
    }

    result.state = std::move(state);
    return result;
}

} // namespace vvm
