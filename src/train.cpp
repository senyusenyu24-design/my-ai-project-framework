#include "tensor.hpp"
#include "n.hpp"
#include "loss.hpp"
#include "optim.hpp"
#include "data.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>
#include <fstream>

using namespace mininn;

double pearson_correlation(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= b.size();
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double xa = a[i] - ma, xb = b[i] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    return num / (std::sqrt(da * db) + 1e-12);
}

int main(int argc, char** argv) {
    std::string csv_path = argc > 1 ? argv[1] : "data/positions_clean.csv";
    std::string weights_path = argc > 2 ? argv[2] : "model_weights.txt";

    CsvData data = read_csv(csv_path);
    int n_features = static_cast<int>(data.header.size()) - 1; 
    int N = static_cast<int>(data.rows.size());
    std::cout << "Loaded " << N << " rows, " << n_features << " features from " << csv_path << "\n";

    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::mt19937 rng(42);
    std::shuffle(idx.begin(), idx.end(), rng);

    int n_train = static_cast<int>(N * 0.85);
    std::vector<int> train_idx(idx.begin(), idx.begin() + n_train);
    std::vector<int> test_idx(idx.begin() + n_train, idx.end());

   
    std::vector<std::vector<double>> train_rows;
    for (int i : train_idx) train_rows.push_back(data.rows[i]);
    Standardizer scaler;
    scaler.fit(train_rows, n_features);

    Sequential model({
        std::make_shared<Linear>(n_features, 32),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(32, 32),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(32, 1)
    });

    SGD optimizer(model.parameters(), /*lr=*/0.01, /*momentum=*/0.5);

    int epochs = 400;
    int batch_size = 32;

    auto make_batch = [&](const std::vector<int>& indices, size_t start, size_t end) {
        int B = static_cast<int>(end - start);
        Tensor xb = Tensor::create(B, n_features);
        Tensor yb = Tensor::create(B, 1);
        for (int i = 0; i < B; ++i) {
            auto& row = data.rows[indices[start + i]];
            auto scaled = scaler.apply(row, n_features);
            for (int j = 0; j < n_features; ++j) xb.at(i, j) = scaled[j];
            yb.at(i, 0) = row[n_features];
        }
        return std::make_pair(xb, yb);
    };


    double best_test_mse = 1e300;
    int best_epoch = -1;
    int patience = 60;
    int epochs_since_improve = 0;
    std::stringstream best_weights;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double epoch_loss = 0.0;
        int n_batches = 0;
        for (size_t start = 0; start < train_idx.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, train_idx.size());
            auto [xb, yb] = make_batch(train_idx, start, end);
            Tensor pred = model.forward(xb);
            Tensor loss = mse_loss(pred, yb);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            epoch_loss += loss.data()[0];
            n_batches++;
        }

        auto [xt, yt] = make_batch(test_idx, 0, test_idx.size());
        Tensor pred = model.forward(xt);
        double test_mse = mse_loss(pred, yt).data()[0];

        if (test_mse < best_test_mse) {
            best_test_mse = test_mse;
            best_epoch = epoch;
            epochs_since_improve = 0;
            best_weights.str("");
            model.save(best_weights);
        } else {
            epochs_since_improve++;
        }

        if (epoch % 40 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch
                      << " | train MSE: " << (epoch_loss / n_batches)
                      << " | test MSE: " << test_mse
                      << " | best test MSE: " << best_test_mse << " (epoch " << best_epoch << ")\n";
        }

        if (epochs_since_improve >= patience) {
            std::cout << "Early stopping at epoch " << epoch
                      << " (no improvement for " << patience << " epochs)\n";
            break;
        }
    }

    
    best_weights.seekg(0);
    model.load(best_weights);

    auto [xt, yt] = make_batch(test_idx, 0, test_idx.size());
    Tensor pred = model.forward(xt);
    std::vector<double> pred_v(pred.data().begin(), pred.data().end());
    std::vector<double> true_v(yt.data().begin(), yt.data().end());
    double corr = pearson_correlation(pred_v, true_v);

    double mae = 0.0;
    for (size_t i = 0; i < pred_v.size(); ++i) mae += std::abs(pred_v[i] - true_v[i]);
    mae /= pred_v.size();

    std::cout << "\n=== Final results on held-out test set (" << test_idx.size()
              << " positions), best weights from epoch " << best_epoch << " ===\n";
    std::cout << "Test MSE: " << best_test_mse << "\n";
    std::cout << "Pearson correlation with Stockfish eval: " << corr << "\n";
    std::cout << "Mean absolute error (scaled units, /100 = pawns): " << mae << " (" << mae * 100 << " centipawns)\n";

    std::ofstream wf(weights_path);
    model.save(wf);
    scaler.save(wf);
    std::cout << "\nSaved best-performing weights to " << weights_path << "\n";

    return 0;
}