#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>

// Function to generate a random state based on transition probabilities
int generateState(const std::vector<double>& probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

    return distribution(gen);
}

// Function to simulate a Markov chain
void simulateMarkovChain(const std::vector<std::vector<double>>& transitionMatrix, int numSteps) {
    int numStates = transitionMatrix.size();
    std::vector<int> stateSequence;

    int currentState = 0;  // Starting state
    stateSequence.push_back(currentState);

    std::cout << "Simulation Results:" << std::endl;
    std::cout << "Step 0: State " << currentState << std::endl;

    // Perform the random walk
    for (int step = 1; step <= numSteps; ++step) {
        const std::vector<double>& probabilities = transitionMatrix[currentState];

        currentState = generateState(probabilities);
        stateSequence.push_back(currentState);

        std::cout << "Step " << step << ": State " << currentState << std::endl;
    }

    // Print the state sequence
    std::cout << "State Sequence: ";
    for (int state : stateSequence) {
        std::cout << state << " ";
    }
    std::cout << std::endl;
}

// Function to calculate the stationary distribution of a Markov chain
std::vector<double> calculateStationaryDistribution(const std::vector<std::vector<double>>& transitionMatrix) {
    int numStates = transitionMatrix.size();
    Eigen::MatrixXd transitionMatrixEigen(numStates, numStates);
    for (int i = 0; i < numStates; ++i) {
        for (int j = 0; j < numStates; ++j) {
            transitionMatrixEigen(i, j) = transitionMatrix[i][j];
        }
    }

    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(transitionMatrixEigen.transpose());
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().real();
    Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().real();

    // Find the eigenvector corresponding to the eigenvalue 1
    int stationaryIndex = -1;
    for (int i = 0; i < eigenValues.size(); ++i) {
        if (std::abs(eigenValues(i) - 1.0) < 1e-8) {
            stationaryIndex = i;
            break;
        }
    }

    std::vector<double> stationaryDistribution(numStates, 0.0);
    if (stationaryIndex != -1) {
        Eigen::VectorXd stationaryVector = eigenVectors.col(stationaryIndex);
        double sum = stationaryVector.sum();
        for (int i = 0; i < numStates; ++i) {
            stationaryDistribution[i] = stationaryVector(i) / sum;
        }
    }

    return stationaryDistribution;
}

// Function to calculate the expected time until absorption for each state
std::vector<double> calculateExpectedTimeToAbsorption(const std::vector<std::vector<double>>& transitionMatrix) {
    int numStates = transitionMatrix.size();
    Eigen::MatrixXd transitionMatrixEigen(numStates, numStates);
    for (int i = 0; i < numStates; ++i) {
        for (int j = 0; j < numStates; ++j) {
            transitionMatrixEigen(i, j) = transitionMatrix[i][j];
        }
    }

    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(numStates, numStates);
    Eigen::MatrixXd absorptionMatrix = transitionMatrixEigen;
    Eigen::VectorXd expectedTimes = Eigen::VectorXd::Zero(numStates);

    while (true) {
        Eigen::MatrixXd prevExpectedTimes = expectedTimes;
        expectedTimes = absorptionMatrix.col(numStates - 1);

        absorptionMatrix = transitionMatrixEigen * absorptionMatrix;
        absorptionMatrix += identityMatrix;

        bool convergence = ((prevExpectedTimes - expectedTimes).norm() < 1e-8);
        if (convergence) {
            break;
        }
    }

    std::vector<double> expectedTimeVector(numStates);
    for (int i = 0; i < numStates; ++i) {
        expectedTimeVector[i] = expectedTimes(i);
    }

    return expectedTimeVector;
}

int main() {
    // Define the transition matrix (example: 3 states)
    std::vector<std::vector<double>> transitionMatrix = {
        {0.2, 0.3, 0.5},
        {0.4, 0.1, 0.5},
        {0.3, 0.4, 0.3}
    };

    // Simulate the Markov chain
    int numSteps = 10;
    auto start = std::chrono::high_resolution_clock::now();
    simulateMarkovChain(transitionMatrix, numSteps);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Elapsed time for markov chain simulation " << elapsed << " ms" << std::endl;

    // Calculate the stationary distribution
    std::vector<double> stationaryDistribution = calculateStationaryDistribution(transitionMatrix);

    std::cout << "Stationary Distribution:" << std::endl;
    for (int i = 0; i < stationaryDistribution.size(); ++i) {
        std::cout << "State " << i << ": " << stationaryDistribution[i] << std::endl;
    }

    // Calculate the expected time until absorption for each state
    std::vector<double> expectedTimes = calculateExpectedTimeToAbsorption(transitionMatrix);

    std::cout << "Expected Time until Absorption:" << std::endl;
    for (int i = 0; i < expectedTimes.size(); ++i) {
        std::cout << "State " << i << ": " << expectedTimes[i] << std::endl;
    }

    return 0;
}


