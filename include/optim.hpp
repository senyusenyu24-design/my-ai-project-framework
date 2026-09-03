#pragma once
#include "tensor.hpp"
#include <vector>

namespace mininn {

class SGD {
public:
    std::vector<Tensor> params;
    double lr;
    double momentum;
    std::vector<std::vector<double>> velocity;

    SGD(std::vector<Tensor> params_, double lr_, double momentum_ = 0.0)
        : params(std::move(params_)), lr(lr_), momentum(momentum_) {
        if (momentum > 0.0) {
            for (auto& p : params) velocity.push_back(std::vector<double>(p.size(), 0.0));
        }
    }

    void zero_grad() {
        for (auto& p : params) p.zero_grad();
    }

    void step() {
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            for (size_t i = 0; i < p.size(); ++i) {
                if (momentum > 0.0) {
                    velocity[pi][i] = momentum * velocity[pi][i] + p.grad()[i];
                    p.data()[i] -= lr * velocity[pi][i];
                } else {
                    p.data()[i] -= lr * p.grad()[i];
                }
            }
        }
    }
};

} 