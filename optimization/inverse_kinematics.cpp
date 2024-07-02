#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>

using namespace std;

const int POPULATION_SIZE = 100;
const int NUM_GENERATIONS = 1000;
const double MUTATION_RATE = 0.01;

const double L1 = 1.0; // Length of the first link
const double L2 = 1.0; // Length of the second link

const double X_TARGET = 1.5; // Target x-coordinate of the end-effector
const double Y_TARGET = 1.0; // Target y-coordinate of the end-effector

const double PI = 3.14159265359;

// Structure to represent a solution (joint angles)
struct Solution {
    double theta1, theta2; // Joint angles
};

// Function to calculate the forward kinematics (end-effector position) for given joint angles
pair<double, double> forwardKinematics(double theta1, double theta2) {
    double x = L1 * cos(theta1) + L2 * cos(theta1 + theta2);
    double y = L1 * sin(theta1) + L2 * sin(theta1 + theta2);
    return {x, y};
}

// Calculate the fitness of a solution (inverse of distance to the target)
double fitness(const Solution &sol) {
    auto [x, y] = forwardKinematics(sol.theta1, sol.theta2);
    double distance = sqrt(pow(x - X_TARGET, 2) + pow(y - Y_TARGET, 2));
    return 1.0 / (1.0 + distance); // Inverse of distance (maximize fitness)
}

// Initialize a random solution (random joint angles)
Solution randomSolution() {
    Solution sol;
    sol.theta1 = (rand() / double(RAND_MAX)) * 2.0 * PI; // Random angle between 0 and 2*pi
    sol.theta2 = (rand() / double(RAND_MAX)) * 2.0 * PI; // Random angle between 0 and 2*pi
    return sol;
}

// Initialize a population of random solutions
vector<Solution> initializePopulation() {
    vector<Solution> population(POPULATION_SIZE);
    for (auto &sol : population) {
        sol = randomSolution();
    }
    return population;
}

// Selection: Tournament selection
Solution selectParent(const vector<Solution> &population) {
    int tournamentSize = 5;
    vector<Solution> selected(tournamentSize);
    for (auto &ind : selected) {
        ind = population[rand() % POPULATION_SIZE];
    }
    return *max_element(selected.begin(), selected.end(), [](const Solution &a, const Solution &b) {
        return fitness(a) < fitness(b);
    });
}

// Crossover: Single-point crossover
Solution crossover(const Solution &parent1, const Solution &parent2) {
    Solution offspring;
    offspring.theta1 = parent1.theta1; // In this simplified example, just copy from one parent
    offspring.theta2 = parent2.theta2; // and the other
    return offspring;
}

// Mutation: Perturb mutation (small random changes)
void mutate(Solution &individual) {
    if (rand() / double(RAND_MAX) < MUTATION_RATE) {
        individual.theta1 += (rand() / double(RAND_MAX) - 0.5) * 0.1; // Small random change
        individual.theta2 += (rand() / double(RAND_MAX) - 0.5) * 0.1; // Small random change
    }
}

int main() {
    srand(time(0));
    vector<Solution> population = initializePopulation();

    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        vector<Solution> newPopulation(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            Solution parent1 = selectParent(population);
            Solution parent2 = selectParent(population);
            Solution offspring = crossover(parent1, parent2);
            mutate(offspring);
            newPopulation[i] = offspring;
        }
        population = newPopulation;

        // Find the best solution in the current population
        double bestFitness = -1.0;
        Solution bestSolution;
        for (const auto &sol : population) {
            double currentFitness = fitness(sol);
            if (currentFitness > bestFitness) {
                bestFitness = currentFitness;
                bestSolution = sol;
            }
        }

        cout << "Generation " << generation << " - Best Fitness: " << bestFitness << endl;
        cout << "Best Solution: theta1 = " << bestSolution.theta1 << ", theta2 = " << bestSolution.theta2 << endl;
    }

    return 0;
}
