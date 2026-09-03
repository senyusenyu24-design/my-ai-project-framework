#pragma once
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace mininn {

class Module {
public:
    virtual Tensor forward(Tensor x) = 0;
    virtual std::vector<Tensor> parameters() { return {}; }
    virtual ~Module() = default;
};

class Linear : public Module {
public:
    Tensor W, b; 
    Linear(int in_features, int out_features) {
        double scale = std::sqrt(2.0 / in_features); // 
        W = Tensor::randn(in_features, out_features, true, scale);
        b = Tensor::zeros(1, out_features, true);
    }
    Tensor forward(Tensor x) override {
        return x.matmul(W) + b;
    }
    std::vector<Tensor> parameters() override { return {W, b}; }
};

class ReLU : public Module {
public:
    Tensor forward(Tensor x) override { return x.relu(); }
};

class Sigmoid : public Module {
public:
    Tensor forward(Tensor x) override { return x.sigmoid(); }
};

class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;
    explicit Sequential(std::vector<std::shared_ptr<Module>> l) : layers(std::move(l)) {}
    Tensor forward(Tensor x) override {
        for (auto& l : layers) x = l->forward(x);
        return x;
    }
    std::vector<Tensor> parameters() override {
        std::vector<Tensor> params;
        for (auto& l : layers) {
            auto p = l->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

} 