#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


void sortrows(std::vector<std::pair<std::vector<float>, int>>& matrix, int col) {    
    std::stable_sort(matrix.begin(),
              matrix.end(),
              [col](const std::pair<std::vector<float>, int>& lhs, const std::pair<std::vector<float>, int>& rhs) {
                  return lhs.first[col] > rhs.first[col];
              });
}

std::vector<std::pair<std::vector<float>, int>> get_copy_with_samples(std::vector<std::vector<float>> &x, std::vector<int> &y, std::vector<int> &samples) {
    std::vector<std::pair<std::vector<float>, int>> data;

    for (int sample : samples) {
        data.push_back(std::make_pair(x[sample], y[sample]));
    }

    return data;
}

class Data {
public:
    Data();
    void loadTrainingData(std::string fn_data, std::string fn_labels);
    void loadTestingData(std::string fn_data, std::string fn_labels);
    std::vector<std::vector<float>> trainData;
    std::vector<std::vector<float>> testData;
    std::vector<int> trainLabels;
    std::vector<int> testLabels;

private:
    void loadData(std::string fn, std::vector<std::vector<float>> &data);
    void loadLabels(std::string fn, std::vector<int> &labels);
};

Data::Data() {
}

void Data::loadData(std::string filename, std::vector<std::vector<float>> &data) {
    std::ifstream infile(filename);

    std::string line = "";
    data.clear();

    while(getline(infile, line)) {
        std::stringstream str_stream(line);
        std::vector<float> dataRow;

        while(str_stream.good()) {
            std::string temp;
            getline(str_stream, temp, ',');
            dataRow.push_back(stof(temp));
        }

        data.push_back(dataRow);
    }
}

void Data::loadLabels(std::string filename, std::vector<int> &labels) {
    std::ifstream infile(filename);

    std::string line = "";
    labels.clear();

    while(getline(infile, line)) {
        std::stringstream str_stream(line);

        while(str_stream.good()) {
            std::string temp;
            getline(str_stream, temp, ',');
            labels.push_back(stoi(temp));
        }
    }
}

void Data::loadTrainingData(std::string fn_data, std::string fn_labels) {
    loadData(fn_data, trainData);
    loadLabels(fn_labels, trainLabels);
}

void Data::loadTestingData(std::string fn_data, std::string fn_labels) {
    loadData(fn_data, testData);
    loadLabels(fn_labels, testLabels);
}

class DecisionTree {
public:
    DecisionTree(std::vector<std::pair<std::vector<float>, int>> &data1,
                int num_all_features1, int num_features1, int num_classes,
                int max_depth1, int depth1, int min_split);
    std::vector<int> predict(std::vector<std::vector<float>> x);

private:
    std::vector<std::pair<std::vector<float>, int>> data;
    std::vector<int> features;
    int num_of_classes;
    int max_depth;
    int depth;
    int min_samples_split;
    bool is_leaf;
    int c;
    int split_feature;
    float split_point;
    float old_gini_impurity;
    DecisionTree *left;
    DecisionTree *right;
    int num_all_features;
    int num_features;

    void fit();
    void findBestSplit(float &best_gini);
    void findBestSplitForFeature(int feature, float &feature_split_point, float &feature_best_gini);
    float giniGain(int feature, float split_val);
    float giniImpurity();
    int predictSample(std::vector<float> sample);
};

DecisionTree::DecisionTree(std::vector<std::pair<std::vector<float>, int>> &data1,
                            int num_all_features1, int num_features1, int num_classes, int _max_depth = std::numeric_limits<int>::max(), 
                            int _depth = 0, int min_split = 2) {

    data = data1;
    num_features = num_features1;
    num_all_features = num_all_features1;
    for (int i = 0; i < num_features; i++)
        features.push_back(rand() % num_all_features);
    num_of_classes = num_classes;
    max_depth = _max_depth;
    depth = _depth;
    min_samples_split = min_split;
    is_leaf = false;
    left = nullptr;
    right = nullptr;
    old_gini_impurity = giniImpurity();
    fit();
}

void DecisionTree::fit() {
    if (depth == max_depth - 1) 
        is_leaf = true;

    float best_gini = -1;
    findBestSplit(best_gini);

    if (best_gini < 0) { 
        is_leaf = true;
    }

    std::vector<std::pair<std::vector<float>, int>> data_left;
    std::vector<std::pair<std::vector<float>, int>> data_right;

    if (!is_leaf) {
        for (int i = 0; i < data.size(); i++) {
            if (data[i].first[split_feature] < split_point)
                data_left.push_back(data[i]);
            else
                data_right.push_back(data[i]);
        }
    }

    if (data_left.size() < min_samples_split || data_right.size() < min_samples_split)
        is_leaf = true;

    if (!is_leaf) {
        left = new DecisionTree(data_left, num_all_features, num_features, num_of_classes, max_depth, depth + 1, min_samples_split);
        right = new DecisionTree(data_right, num_all_features, num_features, num_of_classes, max_depth, depth + 1, min_samples_split);
    } else {
        std::vector<int> class_count(num_of_classes, 0);

        for (int i = 0; i < data.size(); i++) 
            class_count[data[i].second]++;

        c = std::distance(class_count.begin(), 
                            std::max_element(class_count.begin(),
                            class_count.end()));
    }
}

void DecisionTree::findBestSplit(float &best_gini) {
    for (int feature : features) {
        float feature_split_point, feature_best_gini = 0;
        findBestSplitForFeature(feature, feature_split_point, feature_best_gini);
        
        if (feature_best_gini > best_gini) {
            best_gini = feature_best_gini;
            split_point = feature_split_point;
            split_feature = feature;
        }
    }
}

void DecisionTree::findBestSplitForFeature(int feature, float &feature_split_point, float &feature_best_gini)  {
    std::vector<int> k_tiles;
    sortrows(data, feature);
    int num_tiles = std::sqrt(data.size());
    
    for (int i = 0; i < std::min(num_tiles, (int) data.size()); i++) {
        k_tiles.push_back(data.size() * i / num_tiles);
    }
    k_tiles.push_back(data.size() - 1);

    for (int tile : k_tiles) {
        float gini = giniGain(feature, data[tile].first[feature]);
        
        if (gini > feature_best_gini) {
            feature_best_gini = gini;
            feature_split_point = data[tile].first[feature];
        }
    }
}

float DecisionTree::giniGain(int feature, float split_val) {
    std::vector<int> class_count_left(num_of_classes, 0);
    std::vector<int> class_count_right(num_of_classes, 0);

    int count_left = 0, count_right = 0;
    for (int i = 0; i < data.size(); i++) {
        if (data[i].first[feature] < split_val) {
            class_count_left[data[i].second]++;
            count_left++;
        } else {
            class_count_right[data[i].second]++;
            count_right++;
        }
    }

    float gini_impurity_left = 1;
    float gini_impurity_right = 1;

    for (int i = 0; i < class_count_left.size(); i++) {
        float prob_left = (float) class_count_left[i] / count_left;
        float prob_right = (float) class_count_right[i] / count_right;

        gini_impurity_left -= prob_left * prob_left;
        gini_impurity_right -= prob_right * prob_right;
    }

    float gini_impurity = (gini_impurity_left * count_left + gini_impurity_right * count_right) / (count_left + count_right);
    float gini_gain = old_gini_impurity - gini_impurity;

    return gini_gain;
}

float DecisionTree::giniImpurity() {
    std::vector<int> class_count(num_of_classes, 0);

    for (int i = 0; i < data.size(); i++) 
        class_count[data[i].second]++;

    float gini_impurity = 1;

    for (int count : class_count) {
        float p = (float) count / data.size();
        gini_impurity -= p * p;
    }
    
    return gini_impurity;
}

std::vector<int> DecisionTree::predict(std::vector<std::vector<float>> data) {
    std::vector<int> predictions;

    for (int i = 0; i < data.size(); i++) {
        predictions.push_back(predictSample(data[i]));
    }
    return predictions;
}

int DecisionTree::predictSample(std::vector<float> sample) {
    if (is_leaf)
        return c;

    if (sample[split_feature] < split_point) 
        return left->predictSample(sample);
    else 
        return right->predictSample(sample);
}

class RandomForest {
public:
    RandomForest(std::vector<std::vector<float>> &data, std::vector<int> &labels,
                 int num_trees, int max_depth, int min_samples_split);

    std::vector<int> predict(std::vector<std::vector<float>> x);

private:
    std::vector<std::vector<float>> x;
    std::vector<int> y;
    int n_trees;
    int n_features;
    int max_depth;
    int min_samples_split;
    int num_of_classes;
    std::vector<DecisionTree*> trees;

    DecisionTree* createTree();
};

RandomForest::RandomForest(std::vector<std::vector<float>> &data, std::vector<int> &labels,
                           int num_trees, int depth = std::numeric_limits<int>::max(), int min_samples = 2) {
    x = data;
    y = labels;
    n_trees = num_trees;
    max_depth = depth;
    min_samples_split = min_samples;
    num_of_classes = 0;
    std::unordered_set<int> seenClasses;

    for (int i : y) {
        if (seenClasses.find(i) == seenClasses.end()) {
            ++num_of_classes;
            seenClasses.insert(i);
        }
    }

    n_features = (int) std::sqrt(x[0].size());
    
    
    for (int i = 0; i < num_trees; ++i) {    
        trees.push_back(createTree());
    }
}

DecisionTree* RandomForest::createTree() {
    //std::cout << "Creating Decision Tree" << std::endl;

    std::vector<int> samples;
    for (int i = 0; i < x.size(); i++) 
        samples.push_back(std::rand() % x.size());
    
    std::vector<std::pair<std::vector<float>, int>> data1 = get_copy_with_samples(x, y, samples);
    DecisionTree *tree = new DecisionTree(data1, x[0].size(), n_features, num_of_classes, max_depth, 0, min_samples_split);
    return tree;
}

std::vector<int> RandomForest::predict(std::vector<std::vector<float>> data) {
    std::vector<int> predictions(data.size(), 0);
    std::vector<std::vector<int>> tree_predictions;

    for (int i = 0; i < n_trees; i++) {
        tree_predictions.push_back(trees[i]->predict(data));
    }

    int num_samples = data.size();

    for (int i = 0; i < num_samples; i++) {
        std::vector<int> predictions_count(num_of_classes, 0);
        for (int j = 0; j < n_trees; j++) {
            ++predictions_count[tree_predictions[j][i]];
        }
        int pred = std::distance(predictions_count.begin(), 
                    std::max_element(predictions_count.begin(),
                                     predictions_count.end()));
        predictions[i] = pred;
    }

    return predictions;
}

float accuracy(std::vector<int> predicted, std::vector<int> labels) {
    int correct = 0;

    for (int i = 0; i < predicted.size(); i++) {
        if (predicted[i] == labels[i])
            ++correct;
    }

    return (float) correct / predicted.size();
}


void generateConfusionMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, cv::Mat& confusionMatrix, int numClasses) {
    confusionMatrix = cv::Mat::zeros(numClasses, numClasses, CV_32SC1);

    for (size_t i = 0; i < trueLabels.size(); i++) {
        int trueLabel = trueLabels[i];
        int predictedLabel = predictedLabels[i];
        confusionMatrix.at<int>(trueLabel, predictedLabel)++;
    }
}

void visualizeConfusionMatrix(const cv::Mat& confusionMatrix) {
    cv::Mat cmImage;
    cv::Mat confusionMatrixNormalized;
    cv::normalize(confusionMatrix, confusionMatrixNormalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(confusionMatrixNormalized, cmImage, cv::COLORMAP_JET);
    std::string filePath = "../images/confusionmatrix_heatmap.jpg";
    if (cv::imwrite(filePath, cmImage)) {
        std::cout << "Image saved successfully to " << filePath << std::endl;
    } else {
        std::cout << "Failed to save image to " << filePath << std::endl;
    }
}

void generateClassificationReport(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses) {
    std::cout << "Classification Report:\n";

    std::vector<int> truePositives(numClasses, 0);
    std::vector<int> falsePositives(numClasses, 0);
    std::vector<int> falseNegatives(numClasses, 0);
    std::vector<int> supports(numClasses, 0);

    for (size_t i = 0; i < trueLabels.size(); i++) {
        int trueLabel = trueLabels[i];
        int predictedLabel = predictedLabels[i];

        if (predictedLabel == trueLabel) {
            truePositives[trueLabel]++;
        } else {
            falsePositives[predictedLabel]++;
            falseNegatives[trueLabel]++;
        }
        supports[trueLabel]++;
    }

    for (int classLabel = 0; classLabel < numClasses; classLabel++) {
        int tp = truePositives[classLabel];
        int fp = falsePositives[classLabel];
        int fn = falseNegatives[classLabel];
        int support = supports[classLabel];

        double precision = static_cast<double>(tp) / (tp + fp);
        double recall = static_cast<double>(tp) / (tp + fn);
        double f1Score = 2 * precision * recall / (precision + recall);

        std::cout << "Target " << classLabel << ":\n";
        std::cout << "Precision: " << std::fixed << std::setprecision(2) << precision << "\n";
        std::cout << "Recall: " << std::fixed << std::setprecision(2) << recall << "\n";
        std::cout << "F1-Score: " << std::fixed << std::setprecision(2) << f1Score << "\n";
        std::cout << "Support: " << support << "\n\n";
    }
}


    
int main() {
    std::cout << "Loading training data" << std::endl;
    Data data = Data();
    data.loadTestingData("../data/csv/test_x_iris.csv", "../data/csv/test_y_iris.csv");
    data.loadTrainingData("../data/csv/train_x_iris.csv", "../data/csv/train_y_iris.csv");

    int numFeatures = data.trainData[0].size();
    int targets = std::unordered_set<int>(data.trainLabels.begin(), data.trainLabels.end()).size();

    std::cout << "Training started" << std::endl << std::endl;
    RandomForest rf = RandomForest(data.trainData, data.trainLabels, 10);

    std::cout << "Random Forest created" << std::endl;

    std::vector<int> predictions = rf.predict(data.testData);

    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy(predictions, data.testLabels) << std::endl;

    // Generate confusion matrix
    cv::Mat confusionMatrix;
    generateConfusionMatrix(data.testLabels, predictions, confusionMatrix, targets);

    // Visualize confusion matrix
    visualizeConfusionMatrix(confusionMatrix);

    // Generate classification report
    generateClassificationReport(data.testLabels, predictions, targets);

    return 0;
}