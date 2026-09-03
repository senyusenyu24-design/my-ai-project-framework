#include "tensor.hpp"
#include "n.hpp"
#include "loss.hpp"
#include "optim.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <algorithm>

using namespace mininn;

struct Dataset {
    std::vector<double> X; //
    std::vector<int> y;   
    int N;
};

Dataset make_circles(int n_per_class, double noise, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> angle(0.0, 2 * M_PI);
    std::normal_distribution<double> noise_dist(0.0, noise);
    Dataset ds;
    ds.N = n_per_class * 2;
    ds.X.resize(static_cast<size_t>(ds.N) * 2);
    ds.y.resize(ds.N);

    for (int i = 0; i < n_per_class; ++i) {
        double a = angle(gen);
        double r = 1.0;
        ds.X[i * 2 + 0] = r * std::cos(a) + noise_dist(gen);
        ds.X[i * 2 + 1] = r * std::sin(a) + noise_dist(gen);
        ds.y[i] = 0;
    }
    for (int i = 0; i < n_per_class; ++i) {
        int idx = n_per_class + i;
        double a = angle(gen);
        double r = 2.5;
        ds.X[idx * 2 + 0] = r * std::cos(a) + noise_dist(gen);
        ds.X[idx * 2 + 1] = r * std::sin(a) + noise_dist(gen);
        ds.y[idx] = 1;
    }
    return ds;
}

int main() {
    Dataset ds = make_circles(200, 0.15, 123);

    std::vector<int> indices(ds.N);
    for (int i = 0; i < ds.N; ++i) indices[i] = i;
    std::mt19937 rng(7);
    std::shuffle(indices.begin(), indices.end(), rng);

    int n_train = static_cast<int>(ds.N * 0.8);
    std::vector<int> train_idx(indices.begin(), indices.begin() + n_train);
    std::vector<int> test_idx(indices.begin() + n_train, indices.end());

    
    Sequential model({
        std::make_shared<Linear>(2, 16),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(16, 16),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(16, 2)
    });

    SGD optimizer(model.parameters(), /*lr=*/0.5, /*momentum=*/0.9);

    const int epochs = 300;
    const int batch_size = 32;

    std::cout << "Training a 2->16->16->2 MLP on a nonlinear (concentric circles) dataset\n";
    std::cout << "Train samples: " << train_idx.size() << ", Test samples: " << test_idx.size() << "\n\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double epoch_loss = 0.0;
        int n_batches = 0;

        for (size_t start = 0; start < train_idx.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, train_idx.size());
            int B = static_cast<int>(end - start);

            Tensor xb = Tensor::create(B, 2);
            std::vector<int> yb(B);
            for (int i = 0; i < B; ++i) {
                int idx = train_idx[start + i];
                xb.at(i, 0) = ds.X[idx * 2 + 0];
                xb.at(i, 1) = ds.X[idx * 2 + 1];
                yb[i] = ds.y[idx];
            }

            Tensor logits = model.forward(xb);
            Tensor loss = cross_entropy_loss(logits, yb);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.data()[0];
            n_batches++;
        }

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            int B = static_cast<int>(test_idx.size());
            Tensor xt = Tensor::create(B, 2);
            for (int i = 0; i < B; ++i) {
                int idx = test_idx[i];
                xt.at(i, 0) = ds.X[idx * 2 + 0];
                xt.at(i, 1) = ds.X[idx * 2 + 1];
            }
            Tensor logits = model.forward(xt);
            int correct = 0;
            for (int i = 0; i < B; ++i) {
                double best = -1e300; int best_j = -1;
                for (int j = 0; j < 2; ++j) if (logits.at(i, j) > best) { best = logits.at(i, j); best_j = j; }
                if (best_j == ds.y[test_idx[i]]) correct++;
            }
            std::cout << "Epoch " << epoch
                      << " | train loss: " << (epoch_loss / n_batches)
                      << " | test accuracy: " << (100.0 * correct / B) << "%\n";
        }
    }

    return 0;
}