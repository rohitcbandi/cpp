#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <random>

using namespace std;

const int POPULATION_SIZE = 100;
const int NUM_GENERATIONS = 1000;
const double MUTATION_RATE = 0.01;
const int KNAPSACK_CAPACITY = 50;
const int NUM_ITEMS = 20;

// Structure to represent an item
struct Item {
    int value, weight;
};

// Generate random items
vector<Item> generateItems() {
    vector<Item> items(NUM_ITEMS);
    for (auto &item : items) {
        item.value = rand() % 100 + 1;
        item.weight = rand() % 50 + 1;
    }
    return items;
}

// Initialize a random solution (chromosome)
vector<int> randomSolution() {
    vector<int> solution(NUM_ITEMS);
    for (auto &gene : solution) {
        gene = rand() % 2;
    }
    return solution;
}

// Initialize a population of random solutions
vector<vector<int>> initializePopulation() {
    vector<vector<int>> population(POPULATION_SIZE);
    for (auto &individual : population) {
        individual = randomSolution();
    }
    return population;
}

// Calculate the fitness of a solution
int fitness(const vector<int> &solution, const vector<Item> &items) {
    int totalValue = 0, totalWeight = 0;
    for (int i = 0; i < NUM_ITEMS; i++) {
        if (solution[i] == 1) {
            totalValue += items[i].value;
            totalWeight += items[i].weight;
        }
    }
    if (totalWeight > KNAPSACK_CAPACITY) {
        return 0; // Penalty for exceeding capacity
    }
    return totalValue;
}

// Selection: Tournament selection
vector<int> selectParent(const vector<vector<int>> &population, const vector<int> &fitnesses) {
    int tournamentSize = 5;
    vector<int> selected(tournamentSize);
    for (auto &index : selected) {
        index = rand() % POPULATION_SIZE;
    }
    return population[*max_element(selected.begin(), selected.end(), [&fitnesses](int a, int b) {
        return fitnesses[a] < fitnesses[b];
    })];
}

// Crossover: Single-point crossover
vector<int> crossover(const vector<int> &parent1, const vector<int> &parent2) {
    vector<int> offspring(NUM_ITEMS);
    int crossoverPoint = rand() % NUM_ITEMS;
    for (int i = 0; i < crossoverPoint; i++) {
        offspring[i] = parent1[i];
    }
    for (int i = crossoverPoint; i < NUM_ITEMS; i++) {
        offspring[i] = parent2[i];
    }
    return offspring;
}

// Mutation: Flip mutation
void mutate(vector<int> &individual) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        if (rand() / double(RAND_MAX) < MUTATION_RATE) {
            individual[i] = 1 - individual[i]; // Flip the bit
        }
    }
}

// Evaluate the fitness of the population
vector<int> evaluatePopulation(const vector<vector<int>> &population, const vector<Item> &items) {
    vector<int> fitnesses(POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        fitnesses[i] = fitness(population[i], items);
    }
    return fitnesses;
}

int main() {
    srand(time(0));
    vector<Item> items = generateItems();
    vector<vector<int>> population = initializePopulation();
    vector<int> fitnesses = evaluatePopulation(population, items);

    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        vector<vector<int>> newPopulation(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            vector<int> parent1 = selectParent(population, fitnesses);
            vector<int> parent2 = selectParent(population, fitnesses);
            vector<int> offspring = crossover(parent1, parent2);
            mutate(offspring);
            newPopulation[i] = offspring;
        }
        population = newPopulation;
        fitnesses = evaluatePopulation(population, items);

        auto best = max_element(fitnesses.begin(), fitnesses.end());
        cout << "Generation " << generation << " - Best Fitness: " << *best << endl;
    }

    auto best = max_element(fitnesses.begin(), fitnesses.end());
    int bestIndex = distance(fitnesses.begin(), best);
    cout << "Best Solution Fitness: " << *best << endl;
    for (int gene : population[bestIndex]) {
        cout << gene << " ";
    }
    cout << endl;

    return 0;
}
