#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mininn{

    struct TensorImpl {
        std::vector<double> data;
        std::vector<double> grad;
        int rows, cols;
        bool requires_grad;
        
        std::vector<std::shared_ptr<TensorImpl>> parents;

        std::function<void()> backward_fn;

        TensorImpl(int r, int c, bool rg)
            : data(static_cast<size_t>(r) * c, 0.0),
            grad(static_cast<size_t>(r) * c, 0.0),
            rows(r), cols(c), requires_grad(rg) {}
    };

    class Tensor {
        public:
        std::shared_ptr<TensorImpl> impl;

        Tensor() = default;
        
        Tensor(std::shared_ptr<TensorImpl> p) : impl(std::move(p)) {}





    }






}