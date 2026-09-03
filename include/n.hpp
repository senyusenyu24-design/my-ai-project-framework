#pragma once
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace mininn {

class Module {
public:
    virtual Tensor forward(Tensor x) = 0;
    virtual std::vector<Tensor> parameters() { return {}; }
    
    virtual void save(std::ostream&) const {}
    virtual void load(std::istream&) {}
    virtual ~Module() = default;
};

class Linear : public Module {
public:
    Tensor W, b; 
    Linear(int in_features, int out_features) {
        double scale = std::sqrt(2.0 / in_features); 
        W = Tensor::randn(in_features, out_features, true, scale);
        b = Tensor::zeros(1, out_features, true);
    }
    Tensor forward(Tensor x) override {
        return x.matmul(W) + b;
    }
    std::vector<Tensor> parameters() override { return {W, b}; }

    void save(std::ostream& os) const override {
        os << "Linear " << W.rows() << " " << W.cols() << "\n";
        for (double v : W.impl->data) os << v << " ";
        os << "\n";
        for (double v : b.impl->data) os << v << " ";
        os << "\n";
    }
    void load(std::istream& is) override {
        std::string tag; int r, c;
        is >> tag >> r >> c;
        if (tag != "Linear" || r != W.rows() || c != W.cols())
            throw std::runtime_error("Linear::load: shape/tag mismatch in weight file");
        for (auto& v : W.impl->data) is >> v;
        for (auto& v : b.impl->data) is >> v;
    }
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

    void save(std::ostream& os) const override {
        for (auto& l : layers) l->save(os);
    }
    void load(std::istream& is) override {
        for (auto& l : layers) l->load(is);
    }
};

} 