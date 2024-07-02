#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace std;

const int POPULATION_SIZE = 100;
const int NUM_GENERATIONS = 1000;
const double MUTATION_RATE = 0.01;
const int MEMORY_SIZE = 5; // Size of memory (number of previous moves to remember)

// Payoff matrix constants
const double R = 3.0; // Reward for mutual cooperation
const double S = 0.0; // Sucker's payoff
const double T = 5.0; // Temptation to defect
const double P = 1.0; // Punishment for mutual defection

// Structure to represent a strategy (individual in the population)
struct Strategy {
    vector<char> memory; // Memory to store previous moves

    // Constructor to initialize memory with random moves
    Strategy() {
        for (int i = 0; i < MEMORY_SIZE; i++) {
            char move = rand() % 2 == 0 ? 'C' : 'D'; // Random initial move 'C' or 'D'
            memory.push_back(move);
        }
    }
};

// Function to play the Iterated Prisoner's Dilemma game for two strategies
pair<double, double> playGame(const Strategy &strat1, const Strategy &strat2) {
    int rounds = MEMORY_SIZE * 2; // Number of rounds based on memory size
    double payoff1 = 0.0;
    double payoff2 = 0.0;
    
    // Iterate over each round
    for (int i = 0; i < rounds; i++) {
        char move1, move2;
        
        // Determine moves based on memory (cooperate 'C' or defect 'D')
        if (i < strat1.memory.size())
            move1 = strat1.memory[i];
        else
            move1 = 'C'; // Default to cooperate
        
        if (i < strat2.memory.size())
            move2 = strat2.memory[i];
        else
            move2 = 'C'; // Default to cooperate
        
        // Update payoffs based on moves
        if (move1 == 'C' && move2 == 'C') {
            payoff1 += R;
            payoff2 += R;
        } else if (move1 == 'C' && move2 == 'D') {
            payoff1 += S;
            payoff2 += T;
        } else if (move1 == 'D' && move2 == 'C') {
            payoff1 += T;
            payoff2 += S;
        } else { // (move1 == 'D' && move2 == 'D')
            payoff1 += P;
            payoff2 += P;
        }
        
        // Update memory for each strategy
        if (strat1.memory.size() < MEMORY_SIZE)
            strat1.memory.push_back(move2);
        else {
            strat1.memory.erase(strat1.memory.begin());
            strat1.memory.push_back(move2);
        }
        
        if (strat2.memory.size() < MEMORY_SIZE)
            strat2.memory.push_back(move1);
        else {
            strat2.memory.erase(strat2.memory.begin());
            strat2.memory.push_back(move1);
        }
    }
    
    return {payoff1, payoff2};
}

// Calculate the fitness of a strategy against a population of other strategies
double fitness(const Strategy &current, const vector<Strategy> &population) {
    double totalPayoff = 0.0;
    for (const auto &other : population) {
        auto [payoff1, payoff2] = playGame(current, other);
        totalPayoff += payoff1;
    }
    return totalPayoff; // Fitness is total payoff against the population
}

// Initialize a population of random strategies
vector<Strategy> initializePopulation() {
    vector<Strategy> population(POPULATION_SIZE);
    for (auto &strat : population) {
        strat = Strategy(); // Initialize each strategy with random memory
    }
    return population;
}

// Selection: Tournament selection
Strategy selectParent(const vector<Strategy> &population) {
    int tournamentSize = 5;
    vector<Strategy> selected(tournamentSize);
    for (auto &ind : selected) {
        ind = population[rand() % POPULATION_SIZE];
    }
    return *max_element(selected.begin(), selected.end(), [](const Strategy &a, const Strategy &b) {
        return fitness(a) < fitness(b);
    });
}

// Crossover: Uniform crossover
Strategy crossover(const Strategy &parent1, const Strategy &parent2) {
    Strategy offspring;
    for (int i = 0; i < MEMORY_SIZE; i++) {
        if (rand() / double(RAND_MAX) < 0.5)
            offspring.memory.push_back(parent1.memory[i]);
        else
            offspring.memory.push_back(parent2.memory[i]);
    }
    return offspring;
}

// Mutation: Perturb mutation (small random changes)
void mutate(Strategy &individual) {
    for (int i = 0; i < MEMORY_SIZE; i++) {
        if (rand() / double(RAND_MAX) < MUTATION_RATE) {
            individual.memory[i] = individual.memory[i] == 'C' ? 'D' : 'C'; // Flip the move 'C' to 'D' or 'D' to 'C'
        }
    }
}

int main() {
    srand(time(0));
    vector<Strategy> population = initializePopulation();

    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        vector<Strategy> newPopulation(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            Strategy parent1 = selectParent(population);
            Strategy parent2 = selectParent(population);
            Strategy offspring = crossover(parent1, parent2);
            mutate(offspring);
            newPopulation[i] = offspring;
        }
        population = newPopulation;

        // Find the best strategy in the current population
        double bestFitness = -1.0;
        Strategy bestStrategy;
        for (const auto &strat : population) {
            double currentFitness = fitness(strat, population);
            if (currentFitness > bestFitness) {
                bestFitness = currentFitness;
                bestStrategy = strat;
            }
        }

        cout << "Generation " << generation << " - Best Fitness: " << bestFitness << endl;
        cout << "Best Strategy Memory: ";
        for (char move : bestStrategy.memory) {
            cout << move << " ";
        }
        cout << endl;
    }

    return 0;
}
