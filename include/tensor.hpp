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

        static Tensor create(int rows, int cols, bool requires_grad = false){
            return Tensor(std::make_shared<TensorImpl>(rows, cols, requires_grad));
        }

        static Tensor zeros(int rows, int cols, bool requires_grad = false){
            return create(rows, cols, requires_grad);
        }

        static Tensor from_vector(const std::vector<double>& v, int rows, int cols, bool requires_grad = false) {
            Tensor t = create(rows, cols, requires_grad);
            t.impl->data = v;
            return t;
        }

        static Tensor randn(int rows, int cols, bool requires_grad, double scale) {
            static std::mt19937 gen(42);

            std::normal_distribution<double> dist(0.0, 1.0);

            Tensor t = create(rows, cols, requires_grad);

            for (auto& v : t.impl->data) v = dist(gen) * scale;

            return t;
        }

        int rows() const { return impl->rows; }

        int cols() const { return impl->cols; }

        size_t size() const { return impl->data.size(); }

        double& at(int i, int j) { return impl->data[i * cols() + j]; }

        double at(int i, int j) const { return impl->data[i * cols() + j]; }

        std::vector<double>& data() { return impl->data; }

        std::vector<double>& grad() { return impl->grad; }

        const std::vector<double>& data() const { return impl->data; }

        const std::vector<double>& grad() const { return impl->grad; }

        void zero_grad() { std::fill(impl->grad.begin(), impl->grad.end(), 0.0); }

        static void build_topo(TensorImpl* node, std::unordered_set<TensorImpl*>& visited,
                            std::vector<TensorImpl*>& topo) {
            if (visited.count(node)) return;

            visited.insert(node);

            for (auto& p : node->parents) build_topo(p.get(), visited, topo);

            topo.push_back(node);
        }




    }






}