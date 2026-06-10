#include "vvm/model.hpp"

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>

namespace {

void test_dot_product() {
    const float a_values[] = {1.0F, 0.0F};
    const float b_values[] = {0.0F, 1.0F};
    const float c_values[] = {2.0F, 0.0F};

    assert(std::fabs(vvm::dot_product(a_values, b_values)) < 1.0e-6F);
    assert(std::fabs(vvm::dot_product(a_values, c_values) - 2.0F) < 1.0e-6F);
}

void test_run_shape() {
    vvm::Config config{};
    config.state_dim = 16;
    config.num_ops = 32;
    config.top_k = 4;
    config.steps = 3;

    const vvm::Model model(config);
    const std::vector<float> initial_state = model.seeded_state();
    const vvm::RunResult result = model.run(initial_state);

    assert(result.state.size() == config.state_dim);
    assert(result.trace.size() == config.steps);

    for (const vvm::StepTrace& trace : result.trace) {
        assert(trace.retrieval.indices.size() == config.top_k);
        assert(trace.retrieval.weights.size() == config.top_k);
        assert(trace.state_norm > 0.0F);
        assert(trace.activation_mean >= 0.0F);
    }
}

void test_invalid_config() {
    vvm::Config config{};
    config.num_ops = 2;
    config.top_k = 3;

    bool threw = false;
    try {
        const vvm::Model model(config);
        (void)model;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_dot_product();
    test_run_shape();
    test_invalid_config();

    std::cout << "vvm core tests passed\n";
    return 0;
}
