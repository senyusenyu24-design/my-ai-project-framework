
#include "tensor.hpp"
#include "n.hpp"
#include "data.hpp"



#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <fstream>

using namespace mininn;

int main(int argc, char** argv) {
    std::string weights_path = argc > 1 ? argv[1] : "model_weights.txt";
    int n_features = 20; 

    Sequential model({
        std::make_shared<Linear>(n_features, 32),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(32, 32),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(32, 1)
    });

    std::ifstream wf(weights_path);
    if (!wf) {
        std::cerr << "Could not open weights file: " << weights_path << "\n";
        return 1;
    }
    model.load(wf);
    Standardizer scaler;
    scaler.load(wf);

    std::string line;
    if (!std::getline(std::cin, line)) {
        std::cerr << "No input feature vector on stdin.\n";
        return 1;
    }
    std::stringstream ss(line);
    std::vector<double> raw;
    std::string cell;
    while (std::getline(ss, cell, ',')) raw.push_back(std::stod(cell));

    if ((int)raw.size() != n_features) {
        std::cerr << "Expected " << n_features << " features, got " << raw.size() << "\n";
        return 1;
    }

    auto scaled = scaler.apply(raw, n_features);
    Tensor x = Tensor::create(1, n_features);
    for (int j = 0; j < n_features; ++j) x.at(0, j) = scaled[j];

    Tensor pred = model.forward(x);
    double eval_scaled = pred.at(0, 0);
    double eval_cp = eval_scaled * 100.0; 

    std::cout << "Predicted evaluation: " << eval_cp << " centipawns ("
              << (eval_cp / 100.0) << " pawns), White's POV\n";
    return 0;
}