"""
Lab 2 - Weather Prediction
Implementation of decision tree and random forest models.
"""
import numpy as np
import pickle
import os
from data_processing import build_daily_dataset, prepare_training_data
import math
from collections import Counter #For counting class frequencies.
import random

class DecisionTreeNode:
    """Node in a decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold # Threshold value for the split
        self.left = left # Left child node (samples where feature <= threshold)
        self.right = right # Right child node (samples where feature > threshold)
        self.value = value # Predicted value at a leaf node

class DecisionTree:
    """
    Decision Tree classifier implementing entropy-based information gain.
    """
    def __init__(self, max_depth=8, min_samples_split=10, min_info_gain=0.02, ):
        self.max_depth = max_depth# Maximum tree depth
        self.min_samples_split = min_samples_split# Min samples to split
        self.min_info_gain = min_info_gain# Min info gain to split
        self.root = None
        self.feature_importances_ = None# Feature importance scores

    def fit(self, X, y):
        """
        Build the decision tree.

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
        """
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._grow_tree(X, y) # Build the tree recursively

        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (numpy.ndarray): Feature matrix

        Returns:
            numpy.ndarray: Predicted class labels
        """
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        """
        Predict class label for a single sample.

        Args:
            sample (numpy.ndarray): Feature vector

        Returns:
            int: Predicted class label (0 or 1)
        """
        node = self.root
        while node.value is None:  # Not a leaf node
            if sample[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _entropy(self, y):
        """
        Calculate entropy of a target array.

        Args:
            y (numpy.ndarray): Target array

        Returns:
            float: Entropy value
        """
        n_samples = len(y)
        if n_samples == 0:
            return 0.0

        # Count occurrences of each class
        counts = Counter(y)
        entropy = 0.0

        for count in counts.values():
            p = count / n_samples
            entropy -= p * math.log2(p)

        return entropy

    def _information_gain(self, y, y_left, y_right):
        """
        Calculate information gain from a split.

        Args:
            y (numpy.ndarray): Original target array
            y_left (numpy.ndarray): Target array for left split
            y_right (numpy.ndarray): Target array for right split

        Returns:
            float: Information gain
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        if n == 0 or n_left == 0 or n_right == 0:
            return 0.0

        entropy_parent = self._entropy(y)
        entropy_left = self._entropy(y_left)
        entropy_right = self._entropy(y_right)

        # Weighted entropy of children
        weighted_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # Information gain
        info_gain = entropy_parent - weighted_entropy

        return info_gain

    def _best_split(self, X, y, feature_indices=None):
        """
        Find the best split for the data.

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_indices (list): Indices of features to consider for the split

        Returns:
            tuple: (best_feature_idx, best_threshold, best_info_gain, left_indices, right_indices)
        """
        n_samples, n_features = X.shape

        if feature_indices is None:
            feature_indices = list(range(n_features))

        best_info_gain = -1.0
        best_feature_idx = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        # Check each feature
        for feature_idx in feature_indices:
            # Get unique values for the feature
            thresholds = sorted(set(X[:, feature_idx]))

            # Try each threshold
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2

                # Split the data
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # Calculate information gain
                y_left = y[left_indices]
                y_right = y[right_indices]
                info_gain = self._information_gain(y, y_left, y_right)

                # Update best split if this one is better
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_feature_idx, best_threshold, best_info_gain, best_left_indices, best_right_indices

    def _grow_tree(self, X, y, depth=0, feature_indices=None):
        """
        Recursively grow the decision tree.

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            depth (int): Current depth of the tree
            feature_indices (list): Indices of features to consider

        Returns:
            DecisionTreeNode: Root node of the tree
        """
        n_samples, n_features = X.shape

        # Check if all samples have the same class
        if len(set(y)) == 1:
            return DecisionTreeNode(value=y[0])

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            # Return leaf node with majority class
            majority_class = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=majority_class)

        # Find the best split
        best_feature_idx, best_threshold, best_info_gain, left_indices, right_indices = self._best_split(X, y, feature_indices)

        # If no good split is found or information gain is too small, create a leaf node
        if best_feature_idx is None or best_info_gain < self.min_info_gain:
            majority_class = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=majority_class)

        # Update feature importance
        self.feature_importances_[best_feature_idx] += best_info_gain * n_samples

        # Create child nodes
        left_node = self._grow_tree(X[left_indices], y[left_indices], depth + 1, feature_indices)
        right_node = self._grow_tree(X[right_indices], y[right_indices], depth + 1, feature_indices)

        # Return decision node
        return DecisionTreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )

class RandomForest:
    """
    Random Forest classifier.
    """
    def __init__(self, n_trees=20, max_features=3, max_depth=6, min_samples_split=15, min_info_gain=0.0):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Build the random forest.

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
        """
        self.n_features_total = X.shape[1]
        n_samples = X.shape[0]
        self.feature_importances_ = np.zeros(self.n_features_total)

        for _ in range(self.n_trees):
            # Bootstrap sampling (sampling with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Randomly select a subset of features for this tree
            max_features = min(self.max_features, self.n_features_total)
            feature_indices = random.sample(range(self.n_features_total), max_features)

            # Create and train a decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_info_gain=self.min_info_gain
            )
            tree.fit(X_bootstrap, y_bootstrap)

            # Add tree to the forest
            self.trees.append((tree, feature_indices))

            # Update feature importances
            for i, importance in enumerate(tree.feature_importances_):
                if importance > 0:
                    self.feature_importances_[feature_indices[i % len(feature_indices)]] += importance

        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (numpy.ndarray): Feature matrix

        Returns:
            numpy.ndarray: Predicted class labels
        """
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree, _ in self.trees])

        # Transpose to get predictions per sample
        sample_predictions = tree_predictions.T

        # Majority voting
        y_pred = np.array([Counter(pred).most_common(1)[0][0] for pred in sample_predictions])

        return y_pred

def train_models(X_train, y_train, output_dir="models", param_search=True):
    """
    Train decision tree and random forest models for each target.

    Args:
        X_train (numpy.ndarray): Training feature matrix
        y_train (dict): Dictionary of training target vectors
        output_dir (str): Directory to save the trained models
        param_search (bool): Whether to perform parameter search

    Returns:
        dict: Dictionary of trained models
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    models = {}

    # Define target names
    target_names = ['temp_higher', 'above_avg', 'precipitation']

    # Train models for each target
    for target_name in target_names:
        target = y_train[target_name]

        # Decision Tree models
        if param_search:
            # Try different parameters
            best_tree_accuracy = 0.0
            best_tree = None

            for max_depth in [3, 5, 7, 10, 15, None]:
                for min_samples_split in [2, 5, 10]:
                    for min_info_gain in [0.0, 0.01, 0.05]:
                        tree = DecisionTree(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_info_gain=min_info_gain
                        )
                        tree.fit(X_train, target)

                        # Evaluate on training data (could use cross-validation here)
                        y_pred = tree.predict(X_train)
                        accuracy = np.mean(y_pred == target)

                        if accuracy > best_tree_accuracy:
                            best_tree_accuracy = accuracy
                            best_tree = tree
                            print(f"New best tree for {target_name}: {best_tree_accuracy:.4f} "
                                  f"(max_depth={max_depth}, min_samples_split={min_samples_split}, "
                                  f"min_info_gain={min_info_gain})")
        else:
            # Use default parameters
            best_tree = DecisionTree()
            best_tree.fit(X_train, target)
            y_pred = best_tree.predict(X_train)
            best_tree_accuracy = np.mean(y_pred == target)
            print(f"Tree for {target_name}: {best_tree_accuracy:.4f}")

        # Save the best tree model
        models[f'besttree_{target_name}'] = best_tree
        with open(os.path.join(output_dir, f'besttree_{target_name}.pkl'), 'wb') as f:
            pickle.dump(best_tree, f)

        # Random Forest models
        if param_search:
            # Try different parameters
            best_forest_accuracy = 0.0
            best_forest = None

            for n_trees in [5, 10, 15, 20]:
                for max_features in [2, 3, 4]:
                    for max_depth in [3, 5, 7, 10, None]:
                        forest = RandomForest(
                            n_trees=n_trees,
                            max_features=max_features,
                            max_depth=max_depth
                        )
                        forest.fit(X_train, target)

                        # Evaluate on training data
                        y_pred = forest.predict(X_train)
                        accuracy = np.mean(y_pred == target)

                        if accuracy > best_forest_accuracy:
                            best_forest_accuracy = accuracy
                            best_forest = forest
                            print(f"New best forest for {target_name}: {best_forest_accuracy:.4f} "
                                  f"(n_trees={n_trees}, max_features={max_features}, max_depth={max_depth})")
        else:
            # Use default parameters
            best_forest = RandomForest()
            best_forest.fit(X_train, target)
            y_pred = best_forest.predict(X_train)
            best_forest_accuracy = np.mean(y_pred == target)
            print(f"Forest for {target_name}: {best_forest_accuracy:.4f}")

        # Save the best forest model
        models[f'bestforest_{target_name}'] = best_forest
        with open(os.path.join(output_dir, f'bestforest_{target_name}.pkl'), 'wb') as f:
            pickle.dump(best_forest, f)

    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.

    Args:
        models (dict): Dictionary of trained models
        X_test (numpy.ndarray): Test feature matrix
        y_test (dict): Dictionary of test target vectors

    Returns:
        dict: Dictionary of evaluation metrics
    """
    results = {}

    # Define target names
    target_names = ['temp_higher', 'above_avg', 'precipitation']
    model_types = ['besttree', 'bestforest']

    for model_type in model_types:
        for target_name in target_names:
            model_key = f'{model_type}_{target_name}'

            if model_key in models:
                model = models[model_key]
                target = y_test[target_name]

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = np.mean(y_pred == target)

                # Calculate precision, recall, and F1 score
                tp = np.sum((y_pred == 1) & (target == 1))
                fp = np.sum((y_pred == 1) & (target == 0))
                tn = np.sum((y_pred == 0) & (target == 0))
                fn = np.sum((y_pred == 0) & (target == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results[model_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                print(f"Model {model_key}:")
                print(f"  Accuracy:  {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1 Score:  {f1:.4f}")
                print()

    return results

if __name__ == "__main__":
    # Example usage:
    data_dir = "data"
    city_codes = ["ROC", "BUF", "SYR", "ALB", "BOS", "DTW", "CLE", "PIT", "ERI", "ORD", "MSP"]

    # Load and prepare data
    df = build_daily_dataset(data_dir, city_codes)
    X, y = prepare_training_data(df)

    # Split data into training and testing sets (80/20 split)
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train = {target: values[:n_train] for target, values in y.items()}
    y_test = {target: values[n_train:] for target, values in y.items()}

    # Train models
    models = train_models(X_train, y_train, param_search=True)

    # Evaluate models
    results = evaluate_models(models, X_test, y_test)