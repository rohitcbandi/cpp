#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include "gnuplot-iostream.h"

// Define the GradientDescent class
template<typename T>
class GradientDescent {
private:
    const double REPS = 1.0 + std::sqrt(std::numeric_limits<T>::epsilon());
    const double LEPS = 1.0 - std::sqrt(std::numeric_limits<T>::epsilon());
    const double DNOM = 2.0 * std::sqrt(std::numeric_limits<T>::epsilon());

    // Define the partial function for gradient calculation
    std::vector<T> partial(const std::vector<T>& x) {
        int num_dimensions = x.size();
        std::vector<T> gradient(num_dimensions, 0.0);

        std::vector<T> func_val_upper = x;
        std::vector<T> func_val_lower = x;
        for (int i = 0; i < num_dimensions; i++) {
            func_val_upper[i] += REPS * x[i] / DNOM;
            func_val_lower[i] -= LEPS * x[i] / DNOM;
        }

        T cost_upper = cost_function(func_val_upper);
        T cost_lower = cost_function(func_val_lower);

        for (int i = 0; i < num_dimensions; i++) {
            gradient[i] = (cost_upper - cost_lower) / (2 * x[i] / DNOM);
        }

        return gradient;
    }

public:
    // Define the cost function
    T cost_function(const std::vector<T>& x) {
        T cost = 0.0;
        for (T value : x) {
            cost += std::pow(value, 2);
        }
        return cost;
    }

    std::tuple<std::vector<T>, std::vector<T>> gradient_descent(const std::vector<T>& initial_guess,
                                                                 const std::vector<T>& lower_bound,
                                                                 const std::vector<T>& upper_bound,
                                                                 T tolerance,
                                                                 T initial_learning_rate = 0.0001,
                                                                 T decay_factor = 0.1,
                                                                 int patience = 5) {
        int num_dimensions = initial_guess.size();
        std::vector<T> current_solution = initial_guess;
        std::vector<T> cost_history;
        T best_cost = std::numeric_limits<T>::infinity();
        int counter = 0;
        T learning_rate = initial_learning_rate;

        while (true) {
            std::vector<T> gradients = partial(current_solution);

            std::vector<T> new_solution = current_solution;
            for (int i = 0; i < num_dimensions; i++) {
                new_solution[i] -= learning_rate * gradients[i];
                new_solution[i] = std::min(std::max(new_solution[i], lower_bound[i]), upper_bound[i]);
            }

            T solution_change = 0.0;
            for (int i = 0; i < num_dimensions; i++) {
                solution_change += std::pow(new_solution[i] - current_solution[i], 2);
            }
            solution_change = std::sqrt(solution_change);

            if (solution_change < tolerance) {
                break;
            }

            current_solution = new_solution;

            T cost = cost_function(current_solution);
            cost_history.push_back(cost);

            if (cost < best_cost) {
                best_cost = cost;
                counter = 0;
            } else {
                counter++;
                if(counter >= patience) {
                    learning_rate *= decay_factor;
                }
            }
        }

        return std::make_tuple(current_solution, cost_history);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " num_dimensions initial_guess lower_bound upper_bound" << std::endl;
        return 1;
    }

    // Parse command line arguments
    // Set the initial guesses, bounds, and tolerance as vectors
    int num_dimensions = std::stoi(argv[1]);  // Number of dimensions
    std::vector<double> initial_guess(num_dimensions);  // Initial guesses for each dimension
    std::vector<double> lower_bound(num_dimensions);   // Lower bounds for each dimension
    std::vector<double> upper_bound(num_dimensions);  // Upper bounds for each dimension
    double tolerance = 0.000001;                     // Tolerance for convergence

    for (int i = 0; i < num_dimensions; ++i) {
        initial_guess[i] = std::stod(argv[i + 2]);
        lower_bound[i] = std::stod(argv[i + 2 + num_dimensions]);
        upper_bound[i] = std::stod(argv[i + 2 + 2 * num_dimensions]);
    }
    

    // Create an instance of the GradientDescent class
    GradientDescent<double> gd;

    // Run the gradient descent algorithm with adaptive learning rate (RMSProp) and performance-based decay
    auto start = std::chrono::high_resolution_clock::now();
    auto result = gd.gradient_descent(initial_guess, lower_bound, upper_bound, tolerance);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<double> min_solution = std::get<0>(result);
    std::vector<double> cost_history = std::get<1>(result);

    std::cout << "Minimum found: (";
    for (int i = 0; i < num_dimensions; i++) {
        std::cout << min_solution[i];
        if (i < num_dimensions - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
    std::cout << "Cost at minimum: " << gd.cost_function(min_solution) << std::endl;
    std::cout << "Total time for gradient descent: " << elapsed << " ms" << std::endl;

    // Plot the cost function history using Gnuplot
    Gnuplot gp;
    gp << "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n";
    gp << "set output 'cost_function_history_2.png'\n";
    gp << "plot '-' with lines title 'Cost Function History'\n";
    gp.send1d(cost_history);

    return 0;
}

