# CS305 Park University
# Assignment #4 Corrected Code
# Supervised Learning Lab
# By Cyrille Tekam Tiako
# 04 Sep 2024

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
import math

# KNN Learner Definition
class KNN_learner:
    """Lazy learning algorithm for a categorical target and numeric features"""
    def __init__(self, k=1):
        self.k = k
        self.train_data = None
        self.train_labels = None

    def fit(self, train_data, train_labels):
        """Fit the training data"""
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, test_data):
        """Predict the class of test examples"""
        predictions = [self.predict_single(ex) for ex in test_data]
        return predictions

    def predict_single(self, ex):
        """Predict the class for a single test example"""
        neighbors = self.get_neighbors(ex)
        return self.majority_vote(neighbors)

    def get_neighbors(self, ex):
        """Finds k nearest neighbors of the test example."""
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data or labels have not been set.")

        distances = []
        for train_ex, label in zip(self.train_data, self.train_labels):
            dist = self.euclidean_dist(ex, train_ex)
            distances.append((train_ex, dist, label))

        # Sort by distance and return the labels of the nearest k neighbors
        distances.sort(key=lambda x: x[1])
        return [dist[2] for dist in distances[:self.k]]

    def euclidean_dist(self, ex1, ex2):
        """Calculate Euclidean distance between two examples"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(ex1, ex2)))

    def majority_vote(self, neighbors):
        """Get the most common class among the neighbors"""
        counter = Counter(neighbors)
        return counter.most_common(1)[0][0]


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Step 1: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Baseline predictor (Always guess the mode)
    baseline_guess = Counter(y_train).most_common(1)[0][0]
    baseline_predictions = [baseline_guess] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    print(f'Baseline accuracy: {baseline_accuracy:.4f}')

    # Step 3: Decision Tree Learner
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print(f'Decision Tree accuracy: {dt_accuracy:.4f}')

    # Step 4: K-Nearest Neighbors Learner
    k_values = [1, 3, 5, 7]  # K values to test
    for k in k_values:
        knn = KNN_learner(k=k)
        knn.fit(X_train, y_train)
        knn_predictions = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_predictions)
        print(f'KNN accuracy for k={k}: {knn_accuracy:.4f}')

    # Step 5: Cross-validation using K-Fold
    folds = 10
    kf = KFold(n_splits=folds)
    knn_accs: dict[int, list[float]] = {k: [] for k in k_values}

    for train_idx, test_idx in kf.split(X):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        for k in k_values:
            knn = KNN_learner(k=k)
            knn.fit(X_train_cv, y_train_cv)
            knn_predictions_cv = knn.predict(X_test_cv)
            knn_accuracy_cv = accuracy_score(y_test_cv, knn_predictions_cv)
            knn_accs[k].append(knn_accuracy_cv)

    # Print the cross-validation results
    print("\nCross-validation results:")
    for k in k_values:
        print(f'Average KNN accuracy for k={k}: {np.mean(knn_accs[k]):.4f}')


if __name__ == '__main__':
    main()

Output:
Baseline accuracy: 0.9667
Decision Tree accuracy: 0.9667
KNN accuracy for k=1: 0.9667
KNN accuracy for k=3: 0.9667
KNN accuracy for k=5: 0.9667
KNN accuracy for k=7: 0.9667

Cross-validation results:
Average KNN accuracy for k=1: 0.9667
Average KNN accuracy for k=3: 0.9667
Average KNN accuracy for k=5: 0.9667
Average KNN accuracy for k=7: 0.9667
