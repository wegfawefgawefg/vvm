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

std::vector<float> normalize_backward(std::span<const float> normalized,
                                      std::span<const float> pre_normalized,
                                      std::span<const float> grad_normalized) {
    const float norm = l2_norm(pre_normalized);
    std::vector<float> grad(pre_normalized.size(), 0.0F);
    if (norm <= 1.0e-8F) {
        return grad;
    }

    const float projection = dot(normalized, grad_normalized);
    const float inv_norm = 1.0F / norm;
    for (std::size_t i = 0; i < grad.size(); ++i) {
        grad[i] = (grad_normalized[i] - (normalized[i] * projection)) * inv_norm;
    }
    return grad;
}

float decayed(float value, float decay, std::size_t clock) {
    if (value <= 0.0F) {
        return 0.0F;
    }
    return value * std::pow(decay, static_cast<float>(clock));
}

float apply_heat(std::span<float> values, float stddev, std::mt19937& rng) {
    if (stddev <= 0.0F) {
        return 0.0F;
    }

    std::normal_distribution<float> noise(0.0F, stddev);
    float noise_norm_sq = 0.0F;
    for (float& value : values) {
        const float delta = noise(rng);
        value += delta;
        noise_norm_sq += delta * delta;
    }
    return std::sqrt(noise_norm_sq);
}

std::size_t sample_weighted(std::span<const float> weights, std::mt19937& rng) {
    float total = 0.0F;
    for (float weight : weights) {
        total += weight;
    }

    if (total <= 0.0F) {
        return 0;
    }

    std::uniform_real_distribution<float> dist(0.0F, total);
    float needle = dist(rng);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (needle <= weights[i]) {
            return i;
        }
        needle -= weights[i];
    }
    return weights.size() - 1U;
}

float activate(float value, const Config& config) {
    switch (config.activation) {
    case ActivationKind::Relu:
        return std::max(0.0F, value);
    case ActivationKind::LeakyRelu:
        return value >= 0.0F ? value : config.activation_leak * value;
    case ActivationKind::Clamp:
        return std::clamp(value, -1.0F, 1.0F);
    case ActivationKind::Deadzone:
        if (value > config.activation_threshold) {
            return value - config.activation_threshold;
        }
        if (value < -config.activation_threshold) {
            return value + config.activation_threshold;
        }
        return 0.0F;
    }
    return value;
}

float activation_derivative(float value, const Config& config) {
    switch (config.activation) {
    case ActivationKind::Relu:
        return value > 0.0F ? 1.0F : 0.0F;
    case ActivationKind::LeakyRelu:
        return value >= 0.0F ? 1.0F : config.activation_leak;
    case ActivationKind::Clamp:
        return value > -1.0F && value < 1.0F ? 1.0F : 0.0F;
    case ActivationKind::Deadzone:
        return std::fabs(value) > config.activation_threshold ? 1.0F : 0.0F;
    }
    return 1.0F;
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
    if (config_.candidate_count == 0U || config_.candidate_count > config_.num_ops) {
        throw std::invalid_argument("candidate_count must be in [1, num_ops]");
    }
    if (config_.update_scale < 0.0F) {
        throw std::invalid_argument("update_scale must be nonnegative");
    }
    if (config_.input_scale < 0.0F) {
        throw std::invalid_argument("input_scale must be nonnegative");
    }
    if (config_.activation_threshold < 0.0F) {
        throw std::invalid_argument("activation_threshold must be nonnegative");
    }
    if (config_.activation_leak < 0.0F) {
        throw std::invalid_argument("activation_leak must be nonnegative");
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
        normalize_op(op);
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

float Model::prediction_error(std::span<const float> predicted, std::span<const float> observed,
                              std::span<const float> weights) {
    if (predicted.size() != observed.size() || predicted.size() != weights.size()) {
        throw std::invalid_argument("weighted prediction_error requires equal vector sizes");
    }

    float sum = 0.0F;
    float weight_sum = 0.0F;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        if (weights[i] < 0.0F) {
            throw std::invalid_argument("prediction_error weights must be nonnegative");
        }
        const float error = observed[i] - predicted[i];
        sum += weights[i] * error * error;
        weight_sum += weights[i];
    }
    if (weight_sum <= 0.0F) {
        return 0.0F;
    }
    return sum / weight_sum;
}

void apply_observation(Tick& tick, std::span<const float> observed, float curiosity_scale,
                       std::span<const float> weights) {
    if (tick.predicted_state.size() != observed.size()) {
        throw std::invalid_argument(
            "apply_observation requires observed size to match predicted state");
    }
    if (curiosity_scale < 0.0F) {
        throw std::invalid_argument("curiosity_scale must be nonnegative");
    }

    tick.observed_state.assign(observed.begin(), observed.end());
    if (!weights.empty()) {
        if (weights.size() != observed.size()) {
            throw std::invalid_argument("observation weights must match observed state size");
        }
        tick.observation_weights.assign(weights.begin(), weights.end());
        tick.prediction_error =
            Model::prediction_error(tick.predicted_state, tick.observed_state, weights);
    } else {
        tick.observation_weights.clear();
        tick.prediction_error = Model::prediction_error(tick.predicted_state, tick.observed_state);
    }
    tick.reward.curiosity_reward = curiosity_scale * tick.prediction_error;
    tick.reward.total_reward = tick.reward.curiosity_reward;
    if (tick.reward.has_external_reward) {
        tick.reward.total_reward += tick.reward.external_reward;
    }
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

    std::partial_sort(
        scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(config_.candidate_count),
        scores.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    Retrieval retrieval{};
    retrieval.candidate_indices.resize(config_.candidate_count);
    retrieval.candidate_weights.resize(config_.candidate_count);
    retrieval.max_score = scores.front().first;
    const float min_candidate_score = scores[config_.candidate_count - 1U].first;

    float weight_sum = 0.0F;
    for (std::size_t i = 0; i < config_.candidate_count; ++i) {
        retrieval.candidate_indices[i] = scores[i].second;
        retrieval.candidate_weights[i] =
            std::max(scores[i].first - min_candidate_score + 1.0e-6F, 1.0e-6F);
        weight_sum += retrieval.candidate_weights[i];
    }
    for (float& weight : retrieval.candidate_weights) {
        weight /= weight_sum;
    }
    return retrieval;
}

Model::Prediction Model::predict_from_working_state(std::span<const float> working_state) const {
    Retrieval retrieval = retrieve(working_state);
    retrieval.chosen_index = retrieval.candidate_indices.front();
    retrieval.chosen_score = retrieval.max_score;
    const std::span<const float> op_vector(
        op_bank_.data() + (retrieval.chosen_index * config_.state_dim), config_.state_dim);

    std::vector<float> predicted_state(working_state.begin(), working_state.end());
    std::vector<float> pre_activation(config_.state_dim);
    std::vector<float> post_activation(config_.state_dim);
    float activation_sum = 0.0F;
    for (std::size_t i = 0; i < config_.state_dim; ++i) {
        pre_activation[i] = predicted_state[i] + (config_.update_scale * op_vector[i]);
        post_activation[i] = activate(pre_activation[i], config_);
        predicted_state[i] = post_activation[i];
        activation_sum += std::fabs(post_activation[i]);
    }
    normalize_l2(predicted_state);

    return Prediction{
        .state = std::move(predicted_state),
        .pre_activation = std::move(pre_activation),
        .post_activation = std::move(post_activation),
        .retrieval = std::move(retrieval),
        .activation_mean = activation_sum / static_cast<float>(config_.state_dim),
    };
}

Model::Prediction Model::predict_from_working_state(std::span<const float> working_state,
                                                    std::mt19937& rng) const {
    Retrieval retrieval = retrieve(working_state);
    const std::size_t chosen_rank =
        config_.sample_retrieval ? sample_weighted(retrieval.candidate_weights, rng) : 0U;
    retrieval.chosen_index = retrieval.candidate_indices[chosen_rank];
    retrieval.chosen_score = dot_product(
        working_state,
        std::span<const float>(op_bank_.data() + (retrieval.chosen_index * config_.state_dim),
                               config_.state_dim));
    const std::span<const float> op_vector(
        op_bank_.data() + (retrieval.chosen_index * config_.state_dim), config_.state_dim);

    std::vector<float> predicted_state(working_state.begin(), working_state.end());
    std::vector<float> pre_activation(config_.state_dim);
    std::vector<float> post_activation(config_.state_dim);
    float activation_sum = 0.0F;
    for (std::size_t i = 0; i < config_.state_dim; ++i) {
        pre_activation[i] = predicted_state[i] + (config_.update_scale * op_vector[i]);
        post_activation[i] = activate(pre_activation[i], config_);
        predicted_state[i] = post_activation[i];
        activation_sum += std::fabs(post_activation[i]);
    }
    normalize_l2(predicted_state);

    return Prediction{
        .state = std::move(predicted_state),
        .pre_activation = std::move(pre_activation),
        .post_activation = std::move(post_activation),
        .retrieval = std::move(retrieval),
        .activation_mean = activation_sum / static_cast<float>(config_.state_dim),
    };
}

Tick Model::tick(std::vector<float>& state, std::mt19937& rng, std::size_t clock,
                 std::span<const float> input, RewardSignal reward) {
    if (!input.empty() && input.size() != config_.state_dim) {
        throw std::invalid_argument("input size must match model state_dim");
    }

    const float op_heat = decayed(config_.op_heat_stddev, config_.heat_decay, clock);
    std::vector<float> op_heat_l2_by_op = heat_op_bank(op_heat, rng);
    float op_heat_l2_sq = 0.0F;
    for (const float value : op_heat_l2_by_op) {
        op_heat_l2_sq += value * value;
    }
    const float op_heat_l2 = std::sqrt(op_heat_l2_sq);

    std::vector<float> state_before(state.begin(), state.end());
    std::vector<float> working_state(state.begin(), state.end());
    for (std::size_t i = 0; i < input.size(); ++i) {
        working_state[i] += config_.input_scale * input[i];
    }
    std::vector<float> working_pre_state(working_state.begin(), working_state.end());
    normalize_l2(working_state);

    Prediction prediction = predict_from_working_state(working_state, rng);
    state = prediction.state;

    const float state_heat = decayed(config_.state_heat_stddev, config_.heat_decay, clock);
    const float state_heat_l2 = apply_heat(state, state_heat, rng);
    normalize_l2(state);

    const float chosen_prob = [&prediction]() {
        for (std::size_t i = 0; i < prediction.retrieval.candidate_indices.size(); ++i) {
            if (prediction.retrieval.candidate_indices[i] == prediction.retrieval.chosen_index) {
                return prediction.retrieval.candidate_weights[i];
            }
        }
        return 0.0F;
    }();

    Tick tick_result{
        .state_before = std::move(state_before),
        .working_pre_state = std::move(working_pre_state),
        .working_state = std::move(working_state),
        .candidate_indices = std::move(prediction.retrieval.candidate_indices),
        .candidate_probs = std::move(prediction.retrieval.candidate_weights),
        .chosen_op = prediction.retrieval.chosen_index,
        .chosen_prob = chosen_prob,
        .chosen_score = prediction.retrieval.chosen_score,
        .max_score = prediction.retrieval.max_score,
        .pre_activation = std::move(prediction.pre_activation),
        .post_activation = std::move(prediction.post_activation),
        .predicted_state = std::move(prediction.state),
        .observed_state = state,
        .observation_weights = {},
        .activation_mean = prediction.activation_mean,
        .reward = reward,
        .state_heat_stddev = state_heat,
        .op_heat_stddev = op_heat,
        .state_heat_l2 = state_heat_l2,
        .op_heat_l2 = op_heat_l2,
        .op_heat_l2_by_op = std::move(op_heat_l2_by_op),
    };
    apply_observation(tick_result, state, config_.curiosity_scale);
    return tick_result;
}

TrainResult Model::train_window(std::span<const Tick> ticks, TrainConfig train_config) {
    if (train_config.learning_rate < 0.0F) {
        throw std::invalid_argument("learning_rate must be nonnegative");
    }
    if (train_config.recency_decay < 0.0F || train_config.recency_decay > 1.0F) {
        throw std::invalid_argument("recency_decay must be in [0, 1]");
    }
    if (train_config.max_grad_norm < 0.0F) {
        throw std::invalid_argument("max_grad_norm must be nonnegative");
    }
    if (train_config.rejection_scale < 0.0F) {
        throw std::invalid_argument("rejection_scale must be nonnegative");
    }
    if (train_config.rejection_threshold < 0.0F) {
        throw std::invalid_argument("rejection_threshold must be nonnegative");
    }
    if (train_config.rejection_overuse_scale < 0.0F) {
        throw std::invalid_argument("rejection_overuse_scale must be nonnegative");
    }
    if (!train_config.op_usage_counts.empty() &&
        train_config.op_usage_counts.size() != config_.num_ops) {
        throw std::invalid_argument("op_usage_counts size must match num_ops");
    }
    if (ticks.empty()) {
        return TrainResult{};
    }

    std::vector<float> gradients(op_bank_.size(), 0.0F);
    std::vector<std::size_t> op_counts(config_.num_ops, 0U);

    float weighted_loss = 0.0F;
    float weight_sum = 0.0F;
    float error_sum = 0.0F;

    auto add_loss_gradient = [this](const Tick& tick, float recency_weight,
                                    std::vector<float>& grad_pred) {
        if (tick.observation_weights.empty()) {
            for (std::size_t i = 0; i < config_.state_dim; ++i) {
                grad_pred[i] += recency_weight * 2.0F *
                                (tick.predicted_state[i] - tick.observed_state[i]) /
                                static_cast<float>(config_.state_dim);
            }
        } else {
            float observation_weight_sum = 0.0F;
            for (const float weight : tick.observation_weights) {
                observation_weight_sum += weight;
            }
            if (observation_weight_sum > 0.0F) {
                for (std::size_t i = 0; i < config_.state_dim; ++i) {
                    grad_pred[i] += recency_weight * 2.0F * tick.observation_weights[i] *
                                    (tick.predicted_state[i] - tick.observed_state[i]) /
                                    observation_weight_sum;
                }
            }
        }
    };

    auto add_rejection_gradient = [this, &gradients, &train_config](const Tick& tick,
                                                                    float recency_weight) {
        const float rejection_excess = tick.prediction_error - train_config.rejection_threshold;
        if (train_config.rejection_scale <= 0.0F || rejection_excess <= 0.0F ||
            tick.working_state.empty()) {
            return;
        }

        float usage_multiplier = 1.0F;
        if (train_config.rejection_overuse_scale > 0.0F && !train_config.op_usage_counts.empty()) {
            std::size_t total_usage = 0;
            for (const std::size_t count : train_config.op_usage_counts) {
                total_usage += count;
            }
            const float expected_usage = std::max(1.0F, static_cast<float>(total_usage) /
                                                            static_cast<float>(config_.num_ops));
            const float op_usage = static_cast<float>(train_config.op_usage_counts[tick.chosen_op]);
            const float overuse = std::max(0.0F, (op_usage - expected_usage) / expected_usage);
            usage_multiplier = train_config.rejection_overuse_scale * overuse;
        }

        const float rejection =
            recency_weight * train_config.rejection_scale * rejection_excess * usage_multiplier;
        const std::size_t op_offset = tick.chosen_op * config_.state_dim;
        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            gradients[op_offset + i] += rejection * tick.working_state[i];
        }
    };

    for (std::size_t tick_index = 0; tick_index < ticks.size(); ++tick_index) {
        const Tick& tick = ticks[tick_index];
        if (tick.predicted_state.size() != config_.state_dim ||
            tick.observed_state.size() != config_.state_dim ||
            tick.pre_activation.size() != config_.state_dim ||
            tick.post_activation.size() != config_.state_dim || tick.chosen_op >= config_.num_ops) {
            throw std::invalid_argument("tick is incompatible with model config");
        }
        if (!tick.working_state.empty() && tick.working_state.size() != config_.state_dim) {
            throw std::invalid_argument("tick working_state is incompatible with model config");
        }
        if (train_config.backprop_through_state &&
            (tick.state_before.size() != config_.state_dim ||
             tick.working_pre_state.size() != config_.state_dim)) {
            throw std::invalid_argument(
                "tick state history is required for backprop_through_state");
        }
        if (!tick.observation_weights.empty() &&
            tick.observation_weights.size() != config_.state_dim) {
            throw std::invalid_argument(
                "tick observation_weights is incompatible with model config");
        }

        const std::size_t age = ticks.size() - 1U - tick_index;
        const float recency_weight = std::pow(train_config.recency_decay, static_cast<float>(age));
        weighted_loss += recency_weight * tick.prediction_error;
        weight_sum += recency_weight;
        error_sum += tick.prediction_error;
        ++op_counts[tick.chosen_op];
    }

    if (weight_sum <= 0.0F) {
        return TrainResult{
            .op_update_l2_by_op = std::vector<float>(config_.num_ops, 0.0F),
            .tick_count = ticks.size(),
        };
    }

    if (train_config.backprop_through_state) {
        std::vector<float> grad_state_next(config_.state_dim, 0.0F);
        for (std::size_t reverse_index = ticks.size(); reverse_index > 0U; --reverse_index) {
            const std::size_t tick_index = reverse_index - 1U;
            const Tick& tick = ticks[tick_index];
            const std::size_t age = ticks.size() - 1U - tick_index;
            const float recency_weight =
                std::pow(train_config.recency_decay, static_cast<float>(age));

            std::vector<float> grad_pred(config_.state_dim, 0.0F);
            add_loss_gradient(tick, recency_weight, grad_pred);
            for (std::size_t i = 0; i < config_.state_dim; ++i) {
                grad_pred[i] += grad_state_next[i];
            }

            std::vector<float> grad_post =
                normalize_backward(tick.predicted_state, tick.post_activation, grad_pred);

            std::vector<float> grad_working(config_.state_dim, 0.0F);
            const std::size_t op_offset = tick.chosen_op * config_.state_dim;
            for (std::size_t i = 0; i < config_.state_dim; ++i) {
                const float grad_pre =
                    activation_derivative(tick.pre_activation[i], config_) * grad_post[i];
                gradients[op_offset + i] += config_.update_scale * grad_pre;
                grad_working[i] += grad_pre;
            }
            grad_state_next =
                normalize_backward(tick.working_state, tick.working_pre_state, grad_working);
            add_rejection_gradient(tick, recency_weight);
        }
    } else {
        for (std::size_t tick_index = 0; tick_index < ticks.size(); ++tick_index) {
            const Tick& tick = ticks[tick_index];
            const std::size_t age = ticks.size() - 1U - tick_index;
            const float recency_weight =
                std::pow(train_config.recency_decay, static_cast<float>(age));

            std::vector<float> grad_pred(config_.state_dim, 0.0F);
            add_loss_gradient(tick, recency_weight, grad_pred);

            std::vector<float> grad_post =
                normalize_backward(tick.predicted_state, tick.post_activation, grad_pred);

            const std::size_t op_offset = tick.chosen_op * config_.state_dim;
            for (std::size_t i = 0; i < config_.state_dim; ++i) {
                gradients[op_offset + i] += config_.update_scale *
                                            activation_derivative(tick.pre_activation[i], config_) *
                                            grad_post[i];
            }

            add_rejection_gradient(tick, recency_weight);
        }
    }

    std::size_t updated_ops = 0;
    float learning_update_norm_sq = 0.0F;
    std::vector<float> op_update_l2_by_op(config_.num_ops, 0.0F);
    for (std::size_t op = 0; op < config_.num_ops; ++op) {
        if (op_counts[op] == 0U) {
            continue;
        }

        const std::size_t op_offset = op * config_.state_dim;
        std::vector<float> before(config_.state_dim);
        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            before[i] = op_bank_[op_offset + i];
        }

        float scale = 1.0F / weight_sum;
        if (train_config.average_repeated_ops && op_counts[op] > 0U) {
            scale /= static_cast<float>(op_counts[op]);
        }

        float grad_norm_sq = 0.0F;
        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            gradients[op_offset + i] *= scale;
            grad_norm_sq += gradients[op_offset + i] * gradients[op_offset + i];
        }

        const float grad_norm = std::sqrt(grad_norm_sq);
        float clip_scale = 1.0F;
        if (train_config.max_grad_norm > 0.0F && grad_norm > train_config.max_grad_norm) {
            clip_scale = train_config.max_grad_norm / grad_norm;
        }

        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            op_bank_[op_offset + i] -=
                train_config.learning_rate * clip_scale * gradients[op_offset + i];
        }
        normalize_op(op);
        float op_update_norm_sq = 0.0F;
        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            const float delta = op_bank_[op_offset + i] - before[i];
            learning_update_norm_sq += delta * delta;
            op_update_norm_sq += delta * delta;
        }
        op_update_l2_by_op[op] = std::sqrt(op_update_norm_sq);
        ++updated_ops;
    }

    return TrainResult{
        .loss = weighted_loss / weight_sum,
        .mean_prediction_error = error_sum / static_cast<float>(ticks.size()),
        .learning_update_l2 = std::sqrt(learning_update_norm_sq),
        .op_update_l2_by_op = std::move(op_update_l2_by_op),
        .tick_count = ticks.size(),
        .updated_ops = updated_ops,
    };
}

std::vector<float> Model::heat_op_bank(float stddev, std::mt19937& rng) {
    if (stddev <= 0.0F) {
        return {};
    }

    std::vector<float> heat_l2_by_op(config_.num_ops, 0.0F);
    std::normal_distribution<float> noise(0.0F, stddev);
    for (std::size_t op = 0; op < config_.num_ops; ++op) {
        const std::size_t op_offset = op * config_.state_dim;
        float op_heat_norm_sq = 0.0F;
        for (std::size_t i = 0; i < config_.state_dim; ++i) {
            const float delta = noise(rng);
            op_bank_[op_offset + i] += delta;
            op_heat_norm_sq += delta * delta;
        }
        heat_l2_by_op[op] = std::sqrt(op_heat_norm_sq);
        normalize_op(op);
    }
    return heat_l2_by_op;
}

void Model::normalize_op(std::size_t op) {
    if (op >= config_.num_ops) {
        throw std::invalid_argument("op index out of range");
    }

    std::span<float> op_vector(op_bank_.data() + (op * config_.state_dim), config_.state_dim);
    normalize_l2(op_vector);
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
        Tick tick_result = tick(state, rng, step_index);
        Retrieval retrieval{};
        retrieval.candidate_indices = tick_result.candidate_indices;
        retrieval.candidate_weights = tick_result.candidate_probs;
        retrieval.chosen_index = tick_result.chosen_op;
        retrieval.chosen_score = tick_result.chosen_score;
        retrieval.max_score = tick_result.max_score;

        result.trace.push_back(StepTrace{
            .retrieval = std::move(retrieval),
            .state_norm = l2_norm(state),
            .activation_mean = tick_result.activation_mean,
            .prediction_error = tick_result.prediction_error,
            .curiosity_reward = tick_result.reward.curiosity_reward,
            .state_heat_stddev = tick_result.state_heat_stddev,
            .op_heat_stddev = tick_result.op_heat_stddev,
            .state_heat_l2 = tick_result.state_heat_l2,
            .op_heat_l2 = tick_result.op_heat_l2,
        });
    }

    result.state = std::move(state);
    return result;
}

} // namespace vvm
