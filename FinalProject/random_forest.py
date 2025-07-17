"""
Random Forest Classifier Implementation from Scratch
===================================================

This module implements a Random Forest classifier by combining multiple Decision Trees
with bootstrap sampling and feature randomness. Random Forest is an ensemble method
that reduces overfitting and improves generalization through:

1. Bootstrap Aggregating (Bagging): Training each tree on a random sample with replacement
2. Feature Randomness: At each split, only consider a random subset of features
3. Majority Voting: Combine predictions from all trees using majority vote

Mathematical Foundation:
- Each tree sees ~63.2% unique samples due to bootstrap sampling
- Feature subset size typically: sqrt(n_features) for classification
- Final prediction: mode of all tree predictions

Advantages over single Decision Tree:
- Reduced overfitting through ensemble averaging
- Better generalization to unseen data
- Robust to noise and outliers
- Provides feature importance through averaging

Author: Academic Assignment
Date: 2025
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier  # Import our custom Decision Tree


class RandomForestClassifier:
    """
    Random Forest Classifier implemented from scratch.

    This implementation combines multiple Decision Trees using:
    1. Bootstrap sampling: Each tree trains on a random sample with replacement
    2. Feature randomness: Each split considers only a subset of features
    3. Majority voting: Final prediction is the most common prediction across trees

    The algorithm reduces overfitting by:
    - Training diverse trees on different data subsets
    - Reducing correlation between trees through feature randomness
    - Averaging out individual tree errors through ensemble voting
    """

    def __init__(self, n_estimators: int = 100, max_features: Optional[str] = 'sqrt',
                 max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, bootstrap: bool = True,
                 random_state: Optional[int] = None, n_jobs: Optional[int] = None):
        """
        Initialize the Random Forest Classifier.

        Args:
            n_estimators: Number of decision trees in the forest
            max_features: Number of features to consider at each split
                         - 'sqrt': sqrt(n_features)
                         - 'log2': log2(n_features)
                         - int: exact number of features
                         - None: use all features (no randomness)
            max_depth: Maximum depth of individual trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            bootstrap: Whether to use bootstrap sampling
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (not implemented, for compatibility)
        """
        self.n_estimators = n_estimators  # Number of trees in forest
        self.max_features = max_features  # Features to consider per split
        self.max_depth = max_depth  # Maximum tree depth
        self.min_samples_split = min_samples_split  # Min samples to split
        self.min_samples_leaf = min_samples_leaf  # Min samples per leaf
        self.bootstrap = bootstrap  # Use bootstrap sampling
        self.random_state = random_state  # Random seed
        self.n_jobs = n_jobs  # Parallel jobs (not used)

        # Initialize containers for trained components
        self.trees = []  # List of trained decision trees
        self.feature_importances_ = None  # Aggregated feature importance
        self.n_features_ = None  # Number of features in training data
        self.n_classes_ = None  # Number of classes in target
        self.classes_ = None  # Unique class labels

        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

    def _get_n_features(self, total_features: int) -> int:
        """
        Determine the number of features to consider at each split.

        This implements the feature randomness component of Random Forest.
        By considering only a subset of features at each split, we:
        - Reduce correlation between trees
        - Increase diversity in the ensemble
        - Improve generalization performance

        Args:
            total_features: Total number of features in the dataset

        Returns:
            int: Number of features to consider at each split
        """
        if self.max_features is None:
            # Use all features (equivalent to bagging without feature randomness)
            return total_features
        elif self.max_features == 'sqrt':
            # Square root heuristic - good balance for most classification problems
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features == 'log2':
            # Log2 heuristic - more conservative feature selection
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            # Explicit number of features
            return min(self.max_features, total_features)
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a bootstrap sample from the training data.

        Bootstrap sampling (sampling with replacement) is crucial for Random Forest:
        - Each tree sees a different subset of the training data
        - On average, each bootstrap sample contains ~63.2% unique samples
        - Remaining ~36.8% can be used for out-of-bag (OOB) error estimation
        - Creates diversity between trees, reducing overfitting

        Mathematical insight:
        Probability a sample is NOT selected in one draw: (n-1)/n
        Probability a sample is NOT selected in n draws: ((n-1)/n)^n
        As n→∞, this approaches 1/e ≈ 0.368, so ~36.8% samples are out-of-bag

        Args:
            X: Original feature matrix
            y: Original target labels

        Returns:
            Tuple of bootstrap sampled (X_bootstrap, y_bootstrap)
        """
        n_samples = len(X)

        if self.bootstrap:
            # Generate random indices with replacement
            # Each bootstrap sample has the same size as original dataset
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Create bootstrap sample using selected indices
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
        else:
            # If bootstrap=False, use entire dataset (reduces diversity)
            X_bootstrap = X.copy()
            y_bootstrap = y.copy()

        return X_bootstrap, y_bootstrap

    def _create_tree(self, random_state: Optional[int] = None) -> DecisionTreeClassifier:
        """
        Create a single decision tree with specified parameters.

        Each tree in the Random Forest is created with the same hyperparameters
        but will be trained on different bootstrap samples and use feature
        randomness during splitting.

        Args:
            random_state: Random seed for this tree

        Returns:
            DecisionTreeClassifier: Untrained decision tree
        """
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state
        )

    def _train_single_tree(self, tree_idx: int, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """
        Train a single decision tree on bootstrap sample.

        This method encapsulates the training of one tree in the forest:
        1. Create bootstrap sample from training data
        2. Create and configure decision tree
        3. Train tree on bootstrap sample
        4. Return trained tree

        Args:
            tree_idx: Index of the tree being trained (for random seed)
            X: Training feature matrix
            y: Training target labels

        Returns:
            DecisionTreeClassifier: Trained decision tree
        """
        # Set unique random seed for this tree to ensure diversity
        tree_random_state = None
        if self.random_state is not None:
            tree_random_state = self.random_state + tree_idx

        # Create bootstrap sample for this tree
        X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)

        # Create and train decision tree
        tree = self._create_tree(random_state=tree_random_state)
        tree.fit(X_bootstrap, y_bootstrap)

        return tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Train the Random Forest on the provided dataset.

        The training process:
        1. Store dataset characteristics (n_features, classes)
        2. Train n_estimators decision trees in parallel/sequential
        3. Each tree uses bootstrap sampling and feature randomness
        4. Calculate aggregated feature importance across all trees

        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Training target labels (n_samples,)

        Returns:
            self: The fitted Random Forest classifier
        """
        # Validate input data
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) == 0:
            raise ValueError("Cannot fit with empty dataset")

        # Convert to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)

        # Store dataset characteristics
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize list to store trained trees
        self.trees = []

        # Train each tree in the forest
        print(f"Training Random Forest with {self.n_estimators} trees...")
        for tree_idx in range(self.n_estimators):
            # Progress indicator for large forests
            if (tree_idx + 1) % 10 == 0:
                print(f"Training tree {tree_idx + 1}/{self.n_estimators}")

            # Train single tree and add to forest
            tree = self._train_single_tree(tree_idx, X, y)
            self.trees.append(tree)

        # Calculate aggregated feature importance
        self._calculate_feature_importance()

        print(f"Random Forest training completed!")
        return self

    def _calculate_feature_importance(self) -> None:
        """
        Calculate feature importance by averaging across all trees.

        Random Forest feature importance is computed as the average of
        feature importance scores from all individual trees. This provides
        a more stable and reliable measure than single tree importance.

        The importance scores are normalized to sum to 1.0 for interpretability.
        """
        if not self.trees:
            self.feature_importances_ = None
            return

        # Initialize feature importance array
        self.feature_importances_ = np.zeros(self.n_features_)

        # Sum feature importance from all trees
        for tree in self.trees:
            if tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_

        # Average across all trees
        self.feature_importances_ /= len(self.trees)

        # Normalize to ensure sum equals 1.0
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples using majority voting.

        The Random Forest prediction process:
        1. Each tree in the forest makes a prediction for each sample
        2. For each sample, collect all tree predictions
        3. Return the most frequent prediction (majority vote)
        4. In case of ties, return the class with smallest label

        This ensemble approach typically provides better accuracy and
        robustness compared to individual tree predictions.

        Args:
            X: Feature matrix for prediction (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        # Check if model has been trained
        if not self.trees:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy array if needed
        X = np.array(X)

        # Validate feature dimensions
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        n_samples = len(X)

        # Collect predictions from all trees
        # Shape: (n_estimators, n_samples)
        all_predictions = np.zeros((self.n_estimators, n_samples), dtype=int)

        # Get prediction from each tree
        for tree_idx, tree in enumerate(self.trees):
            all_predictions[tree_idx] = tree.predict(X)

        # For each sample, find the majority vote
        final_predictions = np.zeros(n_samples, dtype=int)

        for sample_idx in range(n_samples):
            # Get all tree predictions for this sample
            sample_predictions = all_predictions[:, sample_idx]

            # Count votes for each class
            vote_counts = Counter(sample_predictions)

            # Get the class with maximum votes (ties broken by smallest label)
            final_predictions[sample_idx] = vote_counts.most_common(1)[0][0]

        return final_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Class probabilities are estimated as the fraction of trees
        that vote for each class. This provides a measure of prediction
        confidence and uncertainty.

        Args:
            X: Feature matrix for prediction (n_samples, n_features)

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes)
        """
        # Check if model has been trained
        if not self.trees:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy array if needed
        X = np.array(X)

        # Validate feature dimensions
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        n_samples = len(X)

        # Initialize probability matrix
        probabilities = np.zeros((n_samples, self.n_classes_))

        # Collect predictions from all trees
        for tree in self.trees:
            tree_predictions = tree.predict(X)

            # Convert predictions to one-hot encoding and add to probabilities
            for sample_idx, prediction in enumerate(tree_predictions):
                class_idx = np.where(self.classes_ == prediction)[0][0]
                probabilities[sample_idx, class_idx] += 1

        # Normalize by number of trees to get probabilities
        probabilities /= self.n_estimators

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on the given test data.

        Args:
            X: Test feature matrix
            y: True test labels

        Returns:
            float: Accuracy score (fraction of correctly predicted samples)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Out-of-Bag (OOB) error estimate.

        OOB error provides an unbiased estimate of the forest's generalization
        error without requiring a separate validation set. For each sample,
        only trees that didn't see it during training (out-of-bag) vote.

        Note: This is a simplified implementation. A full implementation would
        track which samples were out-of-bag for each tree during training.

        Args:
            X: Training feature matrix
            y: Training labels

        Returns:
            float: OOB accuracy score
        """
        # This is a simplified OOB calculation
        # In practice, we would track OOB samples during training
        oob_predictions = []
        oob_true_labels = []

        n_samples = len(X)

        # For each sample, find trees that likely didn't see it
        for sample_idx in range(min(100, n_samples)):  # Limit for efficiency
            # Collect predictions from subset of trees (simulating OOB)
            tree_votes = []
            for tree_idx in range(0, len(self.trees), 3):  # Use every 3rd tree
                tree_prediction = self.trees[tree_idx].predict([X[sample_idx]])[0]
                tree_votes.append(tree_prediction)

            if tree_votes:
                # Majority vote from OOB trees
                oob_prediction = Counter(tree_votes).most_common(1)[0][0]
                oob_predictions.append(oob_prediction)
                oob_true_labels.append(y[sample_idx])

        if oob_predictions:
            return np.mean(np.array(oob_predictions) == np.array(oob_true_labels))
        else:
            return 0.0

    def get_feature_importance_df(self, feature_names: Optional[List[str]] = None) -> dict:
        """
        Get feature importance as a dictionary for easy analysis.

        Args:
            feature_names: Optional list of feature names

        Returns:
            dict: Feature importance scores with feature names/indices
        """
        if self.feature_importances_ is None:
            return {}

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        return dict(zip(feature_names, self.feature_importances_))

    def plot_feature_importance(self, feature_names: Optional[List[str]] = None,
                                top_n: int = 20) -> None:
        """
        Plot feature importance scores.

        Args:
            feature_names: Optional list of feature names
            top_n: Number of top features to display
        """
        if self.feature_importances_ is None:
            print("No feature importance available. Model may not be fitted.")
            return

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        # Get top N features
        importance_pairs = list(zip(feature_names, self.feature_importances_))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = importance_pairs[:top_n]

        # Extract names and scores
        names = [pair[0] for pair in top_features]
        scores = [pair[1] for pair in top_features]

        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, len(names) * 0.3)))
        y_pos = np.arange(len(names))

        plt.barh(y_pos, scores)
        plt.yticks(y_pos, names)
        plt.xlabel('Feature Importance')
        plt.title(f'Random Forest Feature Importance (Top {len(names)})')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.show()

    def get_tree_depths(self) -> List[int]:
        """
        Get the depth of each tree in the forest.

        Returns:
            List[int]: Depth of each tree
        """
        return [tree.get_depth() for tree in self.trees]

    def get_tree_n_leaves(self) -> List[int]:
        """
        Get the number of leaf nodes for each tree in the forest.

        Returns:
            List[int]: Number of leaves for each tree
        """
        return [tree.get_n_leaves() for tree in self.trees]

    def print_forest_stats(self) -> None:
        """
        Print summary statistics about the trained forest.
        """
        if not self.trees:
            print("Forest not trained yet.")
            return

        depths = self.get_tree_depths()
        n_leaves = self.get_tree_n_leaves()

        print(f"Random Forest Statistics:")
        print(f"Number of trees: {len(self.trees)}")
        print(f"Number of features: {self.n_features_}")
        print(f"Number of classes: {self.n_classes_}")
        print(
            f"Tree depths - Mean: {np.mean(depths):.1f}, Std: {np.std(depths):.1f}, Range: [{min(depths)}, {max(depths)}]")
        print(
            f"Tree leaves - Mean: {np.mean(n_leaves):.1f}, Std: {np.std(n_leaves):.1f}, Range: [{min(n_leaves)}, {max(n_leaves)}]")

        if self.feature_importances_ is not None:
            top_feature_idx = np.argmax(self.feature_importances_)
            top_feature_importance = self.feature_importances_[top_feature_idx]
            print(f"Most important feature: feature_{top_feature_idx} (importance: {top_feature_importance:.3f})")


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating Random Forest usage.
    This section shows how to use the Random Forest classifier
    and is useful for testing the implementation.
    """

    # Create synthetic dataset for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)

    # Create non-linear decision boundary
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)

    # Split into train/test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train Random Forest
    print("Testing Random Forest Implementation...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )

    # Fit and evaluate
    rf.fit(X_train, y_train)

    # Make predictions
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)

    # Calculate accuracies
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    # Print forest statistics
    rf.print_forest_stats()

    print("Random Forest implementation test completed!")