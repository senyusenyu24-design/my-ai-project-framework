#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace mininn {

struct CsvData {
    std::vector<std::string> header;
    std::vector<std::vector<double>> rows; 
};

inline CsvData read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("could not open CSV file: " + path);
    CsvData data;
    std::string line;

    std::getline(f, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::stringstream hs(line);
    std::string cell;
    while (std::getline(hs, cell, ',')) data.header.push_back(cell);

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        std::stringstream ss(line);
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        data.rows.push_back(row);
    }
    return data;
}
struct Standardizer {
    std::vector<double> mean, stdev;

    void fit(const std::vector<std::vector<double>>& rows, int n_features) {
        mean.assign(n_features, 0.0);
        stdev.assign(n_features, 0.0);
        for (auto& r : rows)
            for (int j = 0; j < n_features; ++j) mean[j] += r[j];
        for (auto& m : mean) m /= rows.size();

        for (auto& r : rows)
            for (int j = 0; j < n_features; ++j) {
                double d = r[j] - mean[j];
                stdev[j] += d * d;
            }
        for (auto& s : stdev) {
            s = std::sqrt(s / rows.size());
            if (s < 1e-8) s = 1.0; 
        }
    }

    std::vector<double> apply(const std::vector<double>& row, int n_features) const {
        std::vector<double> out(n_features);
        for (int j = 0; j < n_features; ++j) out[j] = (row[j] - mean[j]) / stdev[j];
        return out;
    }

    void save(std::ostream& os) const {
        os << mean.size() << "\n";
        for (double v : mean) os << v << " ";
        os << "\n";
        for (double v : stdev) os << v << " ";
        os << "\n";
    }
    void load(std::istream& is) {
        size_t n; is >> n;
        mean.resize(n); stdev.resize(n);
        for (auto& v : mean) is >> v;
        for (auto& v : stdev) is >> v;
    }
};

} 