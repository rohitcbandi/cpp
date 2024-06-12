#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include "gnuplot-iostream.h"


namespace np {
    template<typename T>
    std::vector<T> linspace(T start, T end, size_t num) {
        std::vector<T> result;
        T step = (end - start) / static_cast<T>(num - 1);
        for (size_t i = 0; i < num; ++i) {
            result.push_back(start + step * static_cast<T>(i));
        }
        return result;
    }

    template<typename T>
    T quantile(const std::vector<T>& data, double q) {
        size_t index = static_cast<size_t>(q * (data.size() - 1));
        std::vector<T> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        return sorted_data[index];
    }

    template <typename T>
    const T& clamp(const T& value, const T& minValue, const T& maxValue) {
        return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}
}

int main() {
    Gnuplot gp;

    std::vector<double> x = np::linspace(0.0, 10.0, 100);
    double lb = np::quantile(x, 0.4);
    double ub = np::quantile(x, 0.6);

    std::vector<double> y_orig = x;
    std::vector<double> y_hard = y_orig;
    for (double& val : y_hard) {
        val = np::clamp(val, lb, ub);
    }

    std::vector<double> y_soft = y_orig;
    for (size_t i = 0; i < y_soft.size(); ++i) {
        if (y_soft[i] > ub) {
            double inside = (1.0 / (0.01 * ub)) * (x[i] - ub) + 1.0;
            y_soft[i] = ub * std::pow(inside, 0.01);
        } else if (y_soft[i] < lb) {
            double inside = (-1.0) * (1.0 / (0.01 * lb)) * (x[i] - lb) + 1.0;
            y_soft[i] = 2.0 * lb - lb * std::pow(inside, 0.01);
        }
    }

    gp << "set terminal qt size 800,800" << std::endl;
    gp << "set xlabel 'x'" << std::endl;
    gp << "set ylabel 'y'" << std::endl;
    gp << "plot '-' title 'original' with lines linestyle 1, '-' title 'hard' with lines linestyle 2";
    for (double alpha : {0.01, 0.05, 0.1, 0.25}) {
        gp << ", '-' title 'alpha = " << alpha << "' with lines linestyle 3";
    }
    gp << std::endl;

    gp.send1d(std::make_tuple(x, y_orig));
    gp.send1d(std::make_tuple(x, y_hard));
    for (double alpha : {0.01, 0.05, 0.1, 0.25}) {
        std::vector<double> y = y_soft;
        for (size_t i = 0; i < y.size(); ++i) {
            if (y[i] > ub) {
                double inside = (1.0 / (alpha * ub)) * (x[i] - ub) + 1.0;
                y[i] = ub * std::pow(inside, alpha);
            } else if (y[i] < lb) {
                double inside = (-1.0) * (1.0 / (alpha * lb)) * (x[i] - lb) + 1.0;
                y[i] = 2.0 * lb - lb * std::pow(inside, alpha);
            }
        }
        gp.send1d(std::make_tuple(x, y));
    }

    return 0;
}
