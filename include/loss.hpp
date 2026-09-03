#pragma once
#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mininn {


inline Tensor mse_loss(Tensor pred, Tensor target) {
    Tensor diff = pred - target;
    Tensor sq = diff.mul(diff);
    return sq.mean();
}

inline Tensor cross_entropy_loss(Tensor logits, const std::vector<int>& labels) {
    int B = logits.rows();
    int C = logits.cols();
    if (static_cast<int>(labels.size()) != B) throw std::runtime_error("labels size mismatch");

    Tensor out = Tensor::create(1, 1, logits.impl->requires_grad);
    std::vector<double> softmax(static_cast<size_t>(B) * C);
    double total_loss = 0.0;

    for (int i = 0; i < B; ++i) {
        double maxv = -1e300;
        for (int j = 0; j < C; ++j) maxv = std::max(maxv, logits.at(i, j));
        double sum = 0.0;
        for (int j = 0; j < C; ++j) {
            double e = std::exp(logits.at(i, j) - maxv);
            softmax[i * C + j] = e;
            sum += e;
        }
        for (int j = 0; j < C; ++j) softmax[i * C + j] /= sum;
        total_loss += -std::log(std::max(softmax[i * C + labels[i]], 1e-12));
    }
    out.impl->data[0] = total_loss / B;
    out.impl->parents = {logits.impl};

    auto logitsImpl = logits.impl;
    auto outImpl = out.impl;
    out.impl->backward_fn = [logitsImpl, outImpl, softmax, labels, B, C]() {
        double g = outImpl->grad[0] / B;
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < C; ++j) {
                double indicator = (j == labels[i]) ? 1.0 : 0.0;
                logitsImpl->grad[i * C + j] += g * (softmax[i * C + j] - indicator);
            }
        }
    };
    return out;
}

} 