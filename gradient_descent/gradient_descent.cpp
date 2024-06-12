#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <iomanip>
#include "gnuplot-iostream.h"

const double REPS = 1.0 + std::sqrt(std::numeric_limits<double>::epsilon());
const double LEPS = 1.0 - std::sqrt(std::numeric_limits<double>::epsilon());
const double DNOM = 2.0 * std::sqrt(std::numeric_limits<double>::epsilon());

// Define the cost function
double cost_function(double x) {
    return std::pow(x, 2) + 2 * x + 4;  // Example cost function
}

// Define the partial function for gradient calculation
double partial(double (*cost_function)(double), double x, double REPS, double LEPS, double DNOM) {
    double result = 0.0;
    double func_val = x;

    for (int i = 0; i < 1; i++) {
        func_val = x + REPS * x / DNOM;
        double cost_upper = cost_function(func_val);

        func_val = x - LEPS * x / DNOM;
        double cost_lower = cost_function(func_val);

        result = (cost_upper - cost_lower) / (2 * x / DNOM);
        func_val = x;
    }

    return result;
}

// Define the gradient descent algorithm with adaptive learning rate (RMSProp) and performance-based decay
std::pair<double, std::vector<double>> gradient_descent(double (*cost_function)(double),
                                                       double initial_guess,
                                                       double lower_bound,
                                                       double upper_bound,
                                                       double tolerance,
                                                       double initial_learning_rate = 0.0001,
                                                       double decay_factor = 0.1,
                                                       int patience = 5) {
    double current_solution = initial_guess;
    std::vector<double> cost_history;
    double best_cost = std::numeric_limits<double>::infinity();
    int counter = 0;
    double learning_rate = initial_learning_rate;

    while (true) {
        // Calculate the gradient using the partial function
        double gradient = partial(cost_function, current_solution, REPS, LEPS, DNOM);

        // Update the solution using the gradient descent formula with the learning rate
        double new_solution = current_solution - learning_rate * gradient;

        // Clip the new solution within the bounds
        if (new_solution < lower_bound)
            new_solution = lower_bound;
        if (new_solution > upper_bound)
            new_solution = upper_bound;

        // Check if the change in solution is below the tolerance
        if (std::abs(new_solution - current_solution) < tolerance) {
            break;
        }

        // Update the current solution and track the cost history
        current_solution = new_solution;
        double cost = cost_function(current_solution);
        cost_history.push_back(cost);

        // Check if the cost is improving
        if (cost < best_cost) {
            best_cost = cost;
            counter = 0;
        } else {
            counter++;
            // Decay the learning rate if performance doesn't improve
            if (counter >= patience) {
                learning_rate *= decay_factor;
            }
        }
    }

    return std::make_pair(current_solution, cost_history);
}

int main() {
    // Set the initial guess, bounds, and tolerance
    double initial_guess = -1.1;  // Initial guess for the minimum
    double lower_bound = -5.0;  // Lower bound for each dimension
    double upper_bound = 5.0;  // Upper bound for each dimension
    double tolerance = 0.000001;  // Tolerance for convergence

    // Run the gradient descent algorithm with adaptive learning rate (RMSProp) and performance-based decay
    auto start = std::chrono::high_resolution_clock::now();
    auto result = gradient_descent(cost_function, initial_guess, lower_bound, upper_bound, tolerance);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double minima = result.first;
    std::vector<double> cost_history = result.second;

    std::cout << "Minimum found: " << minima << std::endl;
    std::cout << "Cost at minimum: " << cost_function(minima) << std::endl;
    std::cout << "Total time for gradient descent: " << elapsed << " ns" << std::endl;

    // Plot the cost function history using Gnuplot
    Gnuplot gp;
    gp << "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n";
    gp << "set output 'cost_function_history.png'\n";
    gp << "plot '-' with lines title 'Cost Function History'\n";
    gp.send1d(cost_history);

    return 0;
}
