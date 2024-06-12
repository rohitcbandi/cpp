#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Function to calculate the expected time to absorption
vector<double> expectedTimeToAbsorption(vector<vector<double>>& transitionMatrix, vector<int>& absorbingStates) {
    int numStates = transitionMatrix.size();
    vector<double> expectedTimes(numStates, 0.0);

    // Create the transient matrix Q
    vector<vector<double>> transientMatrix(numStates, vector<double>(numStates, 0.0));
    for (int i = 0; i < numStates; i++) {
        for (int j = 0; j < numStates; j++) {
            if (i != j) {
                transientMatrix[i][j] = transitionMatrix[i][j];
            }
        }
    }

    // Calculate the fundamental matrix (I - Q)
    vector<vector<double>> fundamentalMatrix(numStates, vector<double>(numStates, 0.0));
    for (int i = 0; i < numStates; i++) {
        for (int j = 0; j < numStates; j++) {
            if (i == j) {
                fundamentalMatrix[i][j] = 1.0;
            } else {
                fundamentalMatrix[i][j] = -transientMatrix[i][j];
            }
        }
    }

    // Calculate the inverse of the fundamental matrix
    for (int k = 0; k < numStates; k++) {
        for (int i = 0; i < numStates; i++) {
            if (i != k) {
                for (int j = 0; j < numStates; j++) {
                    if (j != k) {
                        fundamentalMatrix[i][j] -= (fundamentalMatrix[i][k] * fundamentalMatrix[k][j]) / fundamentalMatrix[k][k];
                    }
                }
            }
        }
        for (int i = 0; i < numStates; i++) {
            if (i != k) {
                fundamentalMatrix[i][k] /= -fundamentalMatrix[k][k];
            }
        }
        for (int j = 0; j < numStates; j++) {
            if (j != k) {
                fundamentalMatrix[k][j] /= fundamentalMatrix[k][k];
            }
        }
        fundamentalMatrix[k][k] = 1.0 / fundamentalMatrix[k][k];
    }

    // Calculate the expected times to absorption
    for (int i = 0; i < numStates; i++) {
        double expectedTime = 0.0;
        for (int absorbingState : absorbingStates) {
            expectedTime += fundamentalMatrix[i][absorbingState];
        }
        expectedTimes[i] = expectedTime;
    }

    return expectedTimes;
}

int main() {
    // Transition matrix of the Markov chain
    vector<vector<double>> transitionMatrix = {
        {0.2, 0.3, 0.4, 0.1}, 
        {0.4, 0.1, 0.3, 0.2},
        {0.0, 0.0, 1.0, 0.0}, // absorbing state 1
        {0.0, 0.0, 0.0, 1.0}  // absorbing state 2
    };

    // Absorbing states indices
    vector<int> absorbingStates = {2, 3};

    // Calculate expected times to absorption
    vector<double> expectedTimes = expectedTimeToAbsorption(transitionMatrix, absorbingStates);

    // Print the expected times to absorption for each state
    for (int i = 0; i < expectedTimes.size(); i++) {
        cout << "Expected time to absorption for state " << i << ": " << expectedTimes[i] << endl;
    }

    return 0;
}
