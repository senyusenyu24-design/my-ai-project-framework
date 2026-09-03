#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mininn {

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

    static Tensor create(int rows, int cols, bool requires_grad = false) {
        return Tensor(std::make_shared<TensorImpl>(rows, cols, requires_grad));
    }

    static Tensor zeros(int rows, int cols, bool requires_grad = false) {
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
    void backward() {
        if (impl->data.size() != 1)
            throw std::runtime_error("backward() only supported on scalar (1x1) tensors (e.g. a loss)");
        std::unordered_set<TensorImpl*> visited;
        std::vector<TensorImpl*> topo;
        build_topo(impl.get(), visited, topo);
        std::fill(impl->grad.begin(), impl->grad.end(), 0.0);
        impl->grad[0] = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn) (*it)->backward_fn();
        }
    }

    Tensor operator+(const Tensor& other) const {
        Tensor a = *this;
        Tensor b = other;
        if (a.rows() == b.rows() && a.cols() == b.cols()) {
            Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad || b.impl->requires_grad);
            for (size_t i = 0; i < out.size(); ++i) out.impl->data[i] = a.impl->data[i] + b.impl->data[i];
            out.impl->parents = {a.impl, b.impl};
            auto aImpl = a.impl; auto bImpl = b.impl; auto outImpl = out.impl;
            out.impl->backward_fn = [aImpl, bImpl, outImpl]() {
                for (size_t i = 0; i < outImpl->data.size(); ++i) {
                    aImpl->grad[i] += outImpl->grad[i];
                    bImpl->grad[i] += outImpl->grad[i];
                }
            };
            return out;
        } else if (b.rows() == 1 && b.cols() == a.cols()) {
            Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad || b.impl->requires_grad);
            for (int i = 0; i < a.rows(); ++i)
                for (int j = 0; j < a.cols(); ++j)
                    out.at(i, j) = a.at(i, j) + b.at(0, j);
            out.impl->parents = {a.impl, b.impl};
            auto aImpl = a.impl; auto bImpl = b.impl; auto outImpl = out.impl;
            int R = a.rows(), C = a.cols();
            out.impl->backward_fn = [aImpl, bImpl, outImpl, R, C]() {
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        double g = outImpl->grad[i * C + j];
                        aImpl->grad[i * C + j] += g;
                        bImpl->grad[j] += g;
                    }
                }
            };
            return out;
        } else {
            throw std::runtime_error("shape mismatch in operator+");
        }
    }

    Tensor operator-(const Tensor& other) const {
        if (rows() != other.rows() || cols() != other.cols())
            throw std::runtime_error("shape mismatch in operator-");
        Tensor a = *this; Tensor b = other;
        Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad || b.impl->requires_grad);
        for (size_t i = 0; i < out.size(); ++i) out.impl->data[i] = a.impl->data[i] - b.impl->data[i];
        out.impl->parents = {a.impl, b.impl};
        auto aImpl = a.impl; auto bImpl = b.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, bImpl, outImpl]() {
            for (size_t i = 0; i < outImpl->data.size(); ++i) {
                aImpl->grad[i] += outImpl->grad[i];
                bImpl->grad[i] -= outImpl->grad[i];
            }
        };
        return out;
    }

    Tensor mul(const Tensor& other) const {
        if (rows() != other.rows() || cols() != other.cols())
            throw std::runtime_error("shape mismatch in mul");
        Tensor a = *this, b = other;
        Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad || b.impl->requires_grad);
        for (size_t i = 0; i < out.size(); ++i) out.impl->data[i] = a.impl->data[i] * b.impl->data[i];
        out.impl->parents = {a.impl, b.impl};
        auto aImpl = a.impl; auto bImpl = b.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, bImpl, outImpl]() {
            for (size_t i = 0; i < outImpl->data.size(); ++i) {
                aImpl->grad[i] += outImpl->grad[i] * bImpl->data[i];
                bImpl->grad[i] += outImpl->grad[i] * aImpl->data[i];
            }
        };
        return out;
    }

    Tensor operator*(double scalar) const {
        Tensor a = *this;
        Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad);
        for (size_t i = 0; i < out.size(); ++i) out.impl->data[i] = a.impl->data[i] * scalar;
        out.impl->parents = {a.impl};
        auto aImpl = a.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, outImpl, scalar]() {
            for (size_t i = 0; i < outImpl->data.size(); ++i) aImpl->grad[i] += outImpl->grad[i] * scalar;
        };
        return out;
    }

    Tensor matmul(const Tensor& other) const {
        if (cols() != other.rows()) throw std::runtime_error("shape mismatch in matmul");
        Tensor a = *this, b = other;
        int M = a.rows(), K = a.cols(), N = b.cols();
        Tensor out = Tensor::create(M, N, a.impl->requires_grad || b.impl->requires_grad);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                double av = a.at(i, k);
                if (av == 0.0) continue;
                for (int j = 0; j < N; ++j)
                    out.at(i, j) += av * b.at(k, j);
            }
        out.impl->parents = {a.impl, b.impl};
        auto aImpl = a.impl; auto bImpl = b.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, bImpl, outImpl, M, K, N]() {
           
            for (int i = 0; i < M; ++i)
                for (int k = 0; k < K; ++k) {
                    double s = 0;
                    for (int j = 0; j < N; ++j) s += outImpl->grad[i * N + j] * bImpl->data[k * N + j];
                    aImpl->grad[i * K + k] += s;
                }
        
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < N; ++j) {
                    double s = 0;
                    for (int i = 0; i < M; ++i) s += aImpl->data[i * K + k] * outImpl->grad[i * N + j];
                    bImpl->grad[k * N + j] += s;
                }
        };
        return out;
    }

    Tensor transpose() const {
        Tensor a = *this;
        Tensor out = Tensor::create(a.cols(), a.rows(), a.impl->requires_grad);
        for (int i = 0; i < a.rows(); ++i)
            for (int j = 0; j < a.cols(); ++j)
                out.at(j, i) = a.at(i, j);
        out.impl->parents = {a.impl};
        auto aImpl = a.impl; auto outImpl = out.impl;
        int R = a.rows(), C = a.cols();
        out.impl->backward_fn = [aImpl, outImpl, R, C]() {
            for (int i = 0; i < R; ++i)
                for (int j = 0; j < C; ++j)
                    aImpl->grad[i * C + j] += outImpl->grad[j * R + i];
        };
        return out;
    }

    Tensor relu() const {
        Tensor a = *this;
        Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad);
        for (size_t i = 0; i < out.size(); ++i) out.impl->data[i] = std::max(0.0, a.impl->data[i]);
        out.impl->parents = {a.impl};
        auto aImpl = a.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, outImpl]() {
            for (size_t i = 0; i < outImpl->data.size(); ++i)
                if (aImpl->data[i] > 0) aImpl->grad[i] += outImpl->grad[i];
        };
        return out;
    }

    Tensor sigmoid() const {
        Tensor a = *this;
        Tensor out = Tensor::create(a.rows(), a.cols(), a.impl->requires_grad);
        for (size_t i = 0; i < out.size(); ++i) out.impl->data[i] = 1.0 / (1.0 + std::exp(-a.impl->data[i]));
        out.impl->parents = {a.impl};
        auto aImpl = a.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, outImpl]() {
            for (size_t i = 0; i < outImpl->data.size(); ++i) {
                double s = outImpl->data[i];
                aImpl->grad[i] += outImpl->grad[i] * s * (1 - s);
            }
        };
        return out;
    }

    Tensor sum() const {
        Tensor a = *this;
        Tensor out = Tensor::create(1, 1, a.impl->requires_grad);
        double s = 0; for (double v : a.impl->data) s += v;
        out.impl->data[0] = s;
        out.impl->parents = {a.impl};
        auto aImpl = a.impl; auto outImpl = out.impl;
        out.impl->backward_fn = [aImpl, outImpl]() {
            for (size_t i = 0; i < aImpl->data.size(); ++i) aImpl->grad[i] += outImpl->grad[0];
        };
        return out;
    }

    Tensor mean() const {
        double n = static_cast<double>(size());
        return sum() * (1.0 / n);
    }
};

}