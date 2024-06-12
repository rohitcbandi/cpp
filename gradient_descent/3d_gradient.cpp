#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include "gnuplot-iostream.h"

const double REPS = 1.0 + std::sqrt(std::numeric_limits<double>::epsilon());
const double LEPS = 1.0 - std::sqrt(std::numeric_limits<double>::epsilon());
const double DNOM = 2.0 * std::sqrt(std::numeric_limits<double>::epsilon());

// Define the cost function
double cost_function(double x, double y, double z) {
    return std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2);  // Example cost function
}

// Define the partial function for gradient calculation
std::vector<double> partial(double (*cost_function)(double, double, double), double x, double y, double z, double REPS, double LEPS, double DNOM) {
    std::vector<double> gradient(3, 0.0);

    double func_val_x = x + REPS * x / DNOM;
    double func_val_y = y + REPS * y / DNOM;
    double func_val_z = z + REPS * z / DNOM;
    double cost_upper = cost_function(func_val_x, func_val_y, func_val_z);

    func_val_x = x - LEPS * x / DNOM;
    func_val_y = y - LEPS * y / DNOM;
    func_val_z = z - LEPS * z / DNOM;
    double cost_lower = cost_function(func_val_x, func_val_y, func_val_z);

    gradient[0] = (cost_upper - cost_lower) / (2 * x / DNOM);
    gradient[1] = (cost_upper - cost_lower) / (2 * y / DNOM);
    gradient[2] = (cost_upper - cost_lower) / (2 * z / DNOM);

    x = func_val_x;
    y = func_val_y;
    z = func_val_z;
    

    return gradient;
}

// Define the gradient descent algorithm with adaptive learning rate (RMSProp) and performance-based decay
std::tuple<double, double, double, std::vector<double>> gradient_descent(double (*cost_function)(double, double, double),
                                     double initial_guess_x,
                                     double initial_guess_y,
                                     double initial_guess_z,
                                     double lower_bound,
                                     double upper_bound,
                                     double tolerance,
                                     double initial_learning_rate = 0.0001,
                                     double decay_factor = 0.1,
                                     int patience = 5) {
    double current_solution_x = initial_guess_x;
    double current_solution_y = initial_guess_y;
    double current_solution_z = initial_guess_z;
    std::vector<double> cost_history;
    double best_cost = std::numeric_limits<double>::infinity();
    int counter = 0;
    double learning_rate = initial_learning_rate;

    while (true) {
        // Calculate the gradients using the partial function
       
        std::vector<double> gradients = partial(cost_function, current_solution_x, current_solution_y, current_solution_z, REPS, LEPS, DNOM);
        double gradient_x = gradients[0];
        double gradient_y = gradients[1];
        double gradient_z = gradients[2];

        // Update the solutions using the gradient descent formula with the learning rate
        double new_solution_x = current_solution_x - learning_rate * gradient_x;
        double new_solution_y = current_solution_y - learning_rate * gradient_y;
        double new_solution_z = current_solution_z - learning_rate * gradient_z;

        // Clip the new solutions within the bounds
        new_solution_x = std::min(std::max(new_solution_x, lower_bound), upper_bound);
        new_solution_y = std::min(std::max(new_solution_y, lower_bound), upper_bound);
        new_solution_z = std::min(std::max(new_solution_z, lower_bound), upper_bound);

        // Check if the change in solutions is below the tolerance
        double solution_change = std::sqrt(std::pow(new_solution_x - current_solution_x, 2) +
                                           std::pow(new_solution_y - current_solution_y, 2) +
                                           std::pow(new_solution_z - current_solution_z, 2));
        if (solution_change < tolerance) {
            break;
        }

        // Update the current solutions and track the cost history
        current_solution_x = new_solution_x;
        current_solution_y = new_solution_y;
        current_solution_z = new_solution_z;

        double cost = cost_function(current_solution_x, current_solution_y, current_solution_z);
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

    return std::make_tuple(current_solution_x, current_solution_y, current_solution_z, cost_history);
}

int main() {
    // Set the initial guesses, bounds, and tolerance
    double initial_guess_x = 0.5;  // Initial guess for x
    double initial_guess_y = 0.5;  // Initial guess for y
    double initial_guess_z = 0.5;   // Initial guess for z
    double lower_bound = -5.0;      // Lower bound for each dimension
    double upper_bound = 5.0;       // Upper bound for each dimension
    double tolerance = 0.000001;      // Tolerance for convergence

    // Run the gradient descent algorithm with adaptive learning rate (RMSProp) and performance-based decay
    auto start = std::chrono::high_resolution_clock::now();
    auto result = gradient_descent(cost_function, initial_guess_x, initial_guess_y, initial_guess_z, lower_bound, upper_bound, tolerance);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double min_x = std::get<0>(result);
    double min_y = std::get<1>(result);
    double min_z = std::get<2>(result);
    std::vector<double> cost_history = std::get<3>(result);

    std::cout << "Minimum found: (" << min_x << ", " << min_y << ", " << min_z << ")" << std::endl;
    std::cout << "Cost at minimum: " << cost_function(min_x, min_y, min_z) << std::endl;
    std::cout << "Total time for gradient descent: " << elapsed << " ms" << std::endl;

    // Plot the cost function history using Gnuplot
    Gnuplot gp;
    gp << "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n";
    gp << "set output 'cost_function_history_1.png'\n";
    gp << "plot '-' with lines title 'Cost Function History'\n";
    gp.send1d(cost_history);

    return 0;
}
