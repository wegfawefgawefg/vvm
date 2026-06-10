#pragma once

#include "vvm/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vvm {

enum class VectorRange {
    Signed,
    Nonnegative,
};

enum class TaskKind {
    CopyInput,
    DelayedCopy,
    Linear2,
    Basis4,
    AlternatingBit,
    Xor,
    SineNext,
    Mnist01,
    Mnist,
};

struct TaskSample {
    std::vector<float> input;
    std::vector<float> target;
    std::vector<float> target_weights;
    int label = -1;
    int class_count = 0;
    std::size_t class_offset = 0;
};

struct TaskDataset {
    std::vector<TaskSample> train;
    std::vector<TaskSample> test;
};

struct TaskConfig {
    TaskKind task = TaskKind::CopyInput;
    std::size_t train_samples = 64;
    std::size_t test_samples = 32;
    std::size_t frames_per_sample = 8;
    std::size_t idle_frames_between_samples = 0;
    std::size_t window_size = 8;
    float learning_rate = 0.05F;
    float learning_rate_decay = 1.0F;
    float momentum = 0.0F;
    float recency_decay = 0.97F;
    float max_grad_norm = 1.0F;
    float rejection_scale = 0.0F;
    float rejection_threshold = 0.02F;
    float rejection_decay = 1.0F;
    float rejection_overuse_scale = 0.0F;
    float class_value_scale = 2.0F;
    float class_loss_weight = 0.0F;
    bool backprop_through_state = false;
    VectorRange vector_range = VectorRange::Signed;
    std::string mnist_dir = "resources/mnist";
    std::uint32_t seed = 0x51A7E5U;
};

struct EvalMetrics {
    float loss = 0.0F;
    float nonclass_loss = 0.0F;
    float class_loss = 0.0F;
    float accuracy = 0.0F;
    float balanced_accuracy = 0.0F;
    float mean_class_margin = 0.0F;
    std::size_t accuracy_samples = 0;
    std::vector<std::size_t> label_counts;
    std::vector<std::size_t> prediction_counts;
};

struct LossPoint {
    float train_loss = 0.0F;
    float self_loss = 0.0F;
    float test_loss = 0.0F;
    float test_nonclass_loss = 0.0F;
    float test_class_loss = 0.0F;
    float test_accuracy = 0.0F;
    float test_balanced_accuracy = 0.0F;
    float mean_class_margin = 0.0F;
    float state_heat_l2 = 0.0F;
    float op_heat_l2 = 0.0F;
    float learning_update_l2 = 0.0F;
    float op_selection_entropy = 0.0F;
    float class_route_purity = 0.0F;
    std::size_t accuracy_samples = 0;
    std::size_t updated_ops = 0;
    std::size_t selected_ops = 0;
    std::size_t total_selections = 0;
    std::size_t max_op_selections = 0;
    std::size_t max_op_heat_index = 0;
    std::size_t max_op_train_index = 0;
    float max_op_heat_l2 = 0.0F;
    float max_op_train_l2 = 0.0F;
    std::vector<std::size_t> label_counts;
    std::vector<std::size_t> prediction_counts;
    std::vector<std::size_t> op_selection_counts;
    std::vector<std::size_t> class_op_selection_counts;
    std::vector<float> op_heat_l2_by_op;
    std::vector<float> op_train_l2_by_op;
};

[[nodiscard]] TaskDataset make_task_dataset(const Config& model_config,
                                            const TaskConfig& task_config);
[[nodiscard]] const char* task_name(TaskKind task);
[[nodiscard]] EvalMetrics evaluate_task_metrics(Model& model, std::span<const TaskSample> samples,
                                                const TaskConfig& task_config);
[[nodiscard]] float evaluate_task_loss(Model& model, std::span<const TaskSample> samples,
                                       const TaskConfig& task_config);
[[nodiscard]] LossPoint train_task_epoch(Model& model, std::span<const TaskSample> train_samples,
                                         std::span<const TaskSample> test_samples,
                                         const TaskConfig& task_config, std::size_t epoch);
[[nodiscard]] std::vector<float> neutral_state(std::size_t state_dim);

} // namespace vvm
