#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>

using namespace std;

const int POPULATION_SIZE = 100;
const int NUM_GENERATIONS = 1000;
const double MUTATION_RATE = 0.01;
const int NUM_CITIES = 20;

// Structure to represent a city
struct City {
    int x, y;
};

// Generate random cities
vector<City> generateCities() {
    vector<City> cities(NUM_CITIES);
    for (auto &city : cities) {
        city.x = rand() % 100;
        city.y = rand() % 100;
    }
    return cities;
}

// Function to calculate the distance between two cities
double distance(const City &a, const City &b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Calculate the total distance of the path (solution)
double totalDistance(const vector<City> &cities, const vector<int> &path) {
    double dist = 0.0;
    for (int i = 0; i < path.size() - 1; i++) {
        dist += distance(cities[path[i]], cities[path[i + 1]]);
    }
    dist += distance(cities[path.back()], cities[path.front()]); // Return to the start
    return dist;
}

// Initialize a random path (solution)
vector<int> randomPath() {
    vector<int> path(NUM_CITIES);
    for (int i = 0; i < NUM_CITIES; ++i) {
        path[i] = i;
    }
    random_shuffle(path.begin() + 1, path.end()); // Shuffle except for the first city (starting point)
    return path;
}

// Initialize a population of random paths
vector<vector<int>> initializePopulation() {
    vector<vector<int>> population(POPULATION_SIZE);
    for (auto &individual : population) {
        individual = randomPath();
    }
    return population;
}

// Evaluate the fitness of the population
vector<double> evaluatePopulation(const vector<vector<int>> &population, const vector<City> &cities) {
    vector<double> fitness(POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        fitness[i] = totalDistance(cities, population[i]);
    }
    return fitness;
}

// Selection: Tournament selection
int selectParent(const vector<double> &fitness) {
    int tournamentSize = 5;
    vector<int> selected(tournamentSize);
    for (auto &index : selected) {
        index = rand() % POPULATION_SIZE;
    }
    return *min_element(selected.begin(), selected.end(), [&fitness](int a, int b) {
        return fitness[a] < fitness[b];
    });
}

// Crossover: Ordered crossover
vector<int> crossover(const vector<int> &parent1, const vector<int> &parent2) {
    vector<int> offspring(NUM_CITIES, -1);
    int start = rand() % NUM_CITIES;
    int end = rand() % NUM_CITIES;
    if (start > end) swap(start, end);

    for (int i = start; i <= end; i++) {
        offspring[i] = parent1[i];
    }

    int current = 0;
    for (int i = 0; i < NUM_CITIES; i++) {
        if (find(offspring.begin(), offspring.end(), parent2[i]) == offspring.end()) {
            while (offspring[current] != -1) {
                current++;
            }
            offspring[current] = parent2[i];
        }
    }
    return offspring;
}

// Mutation: Swap mutation
void mutate(vector<int> &individual) {
    for (int i = 0; i < NUM_CITIES; i++) {
        if (rand() / double(RAND_MAX) < MUTATION_RATE) {
            int j = rand() % NUM_CITIES;
            swap(individual[i], individual[j]);
        }
    }
}

int main() {
    srand(time(0));
    vector<City> cities = generateCities();
    vector<vector<int>> population = initializePopulation();
    vector<double> fitness = evaluatePopulation(population, cities);

    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        vector<vector<int>> newPopulation(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            int parent1Index = selectParent(fitness);
            int parent2Index = selectParent(fitness);
            vector<int> parent1 = population[parent1Index];
            vector<int> parent2 = population[parent2Index];
            vector<int> offspring = crossover(parent1, parent2);
            mutate(offspring);
            newPopulation[i] = offspring;
        }
        population = newPopulation;
        fitness = evaluatePopulation(population, cities);

        auto best = min_element(fitness.begin(), fitness.end());
        cout << "Generation " << generation << " - Best Distance: " << *best << endl;
    }

    auto best = min_element(fitness.begin(), fitness.end());
    int bestIndex = distance(fitness.begin(), best);
    cout << "Best Path Distance: " << *best << endl;
    cout << "Best Path Order: ";
    for (int city : population[bestIndex]) {
        cout << city << " ";
    }
    cout << endl;

    return 0;
}
