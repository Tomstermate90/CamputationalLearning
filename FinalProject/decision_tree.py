"""
Decision Tree Classifier Implementation from Scratch
===================================================

This module implements a Decision Tree classifier using the CART (Classification and Regression Trees)
algorithm with Gini impurity as the splitting criterion. The implementation includes:
- Binary splitting at each node
- Gini impurity calculation for node purity assessment
- Recursive tree building with stopping criteria
- Prediction capability for new instances

Mathematical Foundation:
- Gini Impurity: 1 - Σ(p_i)² where p_i is the probability of class i
- Information Gain: Parent_Gini - Weighted_Average(Children_Gini)

Author: Academic Assignment
Date: 2025
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional


class Node:
    """
    Represents a single node in the decision tree.

    Each node can either be:
    1. Internal node: Contains a splitting condition (feature + threshold)
    2. Leaf node: Contains a class prediction

    Attributes:
        feature (int): Index of the feature used for splitting (None for leaf nodes)
        threshold (float): Threshold value for splitting (None for leaf nodes)
        left (Node): Left child node (for values <= threshold)
        right (Node): Right child node (for values > threshold)
        prediction (int): Class prediction for leaf nodes (None for internal nodes)
        samples (int): Number of training samples that reached this node
        gini (float): Gini impurity of samples at this node
    """

    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 prediction: Optional[int] = None, samples: int = 0, gini: float = 0.0):
        """
        Initialize a decision tree node.

        Args:
            feature: Feature index used for splitting (internal nodes only)
            threshold: Splitting threshold (internal nodes only)
            left: Left child node
            right: Right child node
            prediction: Class prediction (leaf nodes only)
            samples: Number of samples at this node
            gini: Gini impurity at this node
        """
        self.feature = feature  # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child (feature <= threshold)
        self.right = right  # Right child (feature > threshold)
        self.prediction = prediction  # Class prediction for leaf nodes
        self.samples = samples  # Number of samples reaching this node
        self.gini = gini  # Gini impurity of this node

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node.

        Returns:
            bool: True if leaf node (has prediction), False if internal node
        """
        return self.prediction is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implemented from scratch using CART algorithm.

    This implementation uses:
    - Gini impurity as the splitting criterion
    - Binary splits (<=, >) for continuous features
    - Majority class voting for leaf node predictions
    - Recursive tree building with configurable stopping criteria

    The algorithm works by:
    1. At each node, evaluate all possible splits across all features
    2. Choose the split that minimizes weighted Gini impurity
    3. Recursively build left and right subtrees
    4. Stop when stopping criteria are met (max_depth, min_samples_split, etc.)
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, random_state: Optional[int] = None):
        """
        Initialize the Decision Tree Classifier.

        Args:
            max_depth: Maximum depth of the tree (prevents overfitting)
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth  # Maximum tree depth
        self.min_samples_split = min_samples_split  # Min samples to split
        self.min_samples_leaf = min_samples_leaf  # Min samples per leaf
        self.random_state = random_state  # Random seed
        self.root = None  # Root node of the tree
        self.feature_importances_ = None  # Feature importance scores

        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

    def _calculate_gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for a set of labels.

        Gini impurity measures how "impure" or mixed the classes are in a dataset.
        Formula: Gini = 1 - Σ(p_i)² where p_i is the proportion of class i

        Interpretation:
        - Gini = 0: All samples belong to the same class (pure)
        - Gini = 0.5: Classes are evenly distributed (most impure for binary)

        Args:
            y: Array of class labels

        Returns:
            float: Gini impurity value between 0 and 1
        """
        if len(y) == 0:  # Handle empty arrays
            return 0.0

        # Count occurrences of each class
        class_counts = Counter(y)
        total_samples = len(y)

        # Calculate probability of each class: p_i = count_i / total
        gini = 1.0
        for class_count in class_counts.values():
            probability = class_count / total_samples  # p_i
            gini -= probability ** 2  # Subtract p_i²

        return gini

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best feature and threshold to split the data.

        This method evaluates all possible splits across all features and selects
        the one that provides the maximum information gain (minimum weighted Gini).

        Algorithm:
        1. For each feature:
           a. Get all unique values as potential thresholds
           b. For each threshold, split data into left (<=) and right (>) groups
           c. Calculate weighted Gini impurity of the split
           d. Track the split with minimum weighted Gini
        2. Return the best feature, threshold, and information gain

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Tuple containing:
            - best_feature (int): Index of the best feature to split on
            - best_threshold (float): Best threshold value for splitting
            - best_gain (float): Information gain from the best split
        """
        n_samples, n_features = X.shape

        # If all samples have the same label, no split can improve purity
        if len(set(y)) == 1:
            return None, None, 0.0

        # Calculate current Gini impurity (before splitting)
        current_gini = self._calculate_gini(y)

        # Initialize variables to track the best split
        best_gain = 0.0  # Best information gain found
        best_feature = None  # Feature index for best split
        best_threshold = None  # Threshold value for best split

        # Evaluate splits for each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]  # Extract feature column

            # Get unique values as potential split points
            # We use unique values to avoid redundant splits
            unique_values = np.unique(feature_values)

            # For each unique value, try it as a threshold
            for threshold in unique_values:
                # Split data based on threshold
                left_mask = feature_values <= threshold  # Boolean mask for left split
                right_mask = feature_values > threshold  # Boolean mask for right split

                # Skip if split results in empty left or right group
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Get labels for left and right splits
                y_left = y[left_mask]
                y_right = y[right_mask]

                # Calculate weighted Gini impurity after split
                n_left = len(y_left)
                n_right = len(y_right)

                # Weighted average of child node impurities
                weighted_gini = (n_left / n_samples) * self._calculate_gini(y_left) + \
                                (n_right / n_samples) * self._calculate_gini(y_right)

                # Calculate information gain: reduction in impurity
                gain = current_gini - weighted_gini

                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.

        This is the core recursive function that constructs the tree by:
        1. Checking stopping criteria (max depth, min samples, pure node)
        2. Finding the best split using _find_best_split()
        3. Creating left and right child nodes recursively
        4. Returning the constructed node

        Stopping criteria:
        - Maximum depth reached
        - Insufficient samples to split
        - All samples have the same class (pure node)
        - No split provides information gain

        Args:
            X: Feature matrix for current node
            y: Labels for current node
            depth: Current depth in the tree

        Returns:
            Node: The root node of the subtree built from this data
        """
        n_samples = len(y)

        # Calculate node statistics
        gini = self._calculate_gini(y)
        most_common_class = Counter(y).most_common(1)[0][0]  # Majority class

        # Create leaf node if stopping criteria are met
        should_stop = (
                depth >= self.max_depth or  # Max depth reached
                n_samples < self.min_samples_split or  # Too few samples to split
                gini == 0.0 or  # Pure node (all same class)
                len(set(y)) == 1  # All samples same class
        )

        if should_stop:
            # Create and return leaf node with majority class prediction
            return Node(prediction=most_common_class, samples=n_samples, gini=gini)

        # Find the best split for this node
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        # If no beneficial split found, create leaf node
        if best_feature is None or best_gain == 0.0:
            return Node(prediction=most_common_class, samples=n_samples, gini=gini)

        # Split the data based on the best split found
        feature_values = X[:, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = feature_values > best_threshold

        # Check minimum samples per leaf constraint
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return Node(prediction=most_common_class, samples=n_samples, gini=gini)

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Create and return internal node
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=n_samples,
            gini=gini
        )

    def _calculate_feature_importance(self, node: Node, total_samples: int) -> None:
        """
        Calculate feature importance scores based on information gain.

        Feature importance is calculated as the weighted information gain
        provided by each feature across all splits in the tree.

        Formula: importance = (samples/total_samples) * information_gain

        Args:
            node: Current node in the tree
            total_samples: Total number of training samples
        """
        if node.is_leaf():
            return

        # Calculate information gain for this split
        # Current node impurity
        current_impurity = node.gini

        # Weighted average of children impurities
        left_weight = node.left.samples / node.samples
        right_weight = node.right.samples / node.samples
        children_impurity = (left_weight * node.left.gini +
                             right_weight * node.right.gini)

        # Information gain from this split
        gain = current_impurity - children_impurity

        # Weight by number of samples reaching this node
        importance = (node.samples / total_samples) * gain

        # Add to feature importance score
        self.feature_importances_[node.feature] += importance

        # Recursively calculate for children
        self._calculate_feature_importance(node.left, total_samples)
        self._calculate_feature_importance(node.right, total_samples)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Train the decision tree classifier.

        This method builds the complete decision tree by calling the recursive
        _build_tree method starting from the root. It also calculates feature
        importance scores.

        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: The fitted classifier
        """
        # Validate input data
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) == 0:
            raise ValueError("Cannot fit with empty dataset")

        # Convert to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)

        # Initialize feature importance scores
        self.feature_importances_ = np.zeros(X.shape[1])

        # Build the decision tree starting from root
        self.root = self._build_tree(X, y, depth=0)

        # Calculate feature importance scores
        self._calculate_feature_importance(self.root, len(X))

        # Normalize feature importance scores to sum to 1
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def _predict_sample(self, x: np.ndarray) -> int:
        """
        Predict the class for a single sample.

        This method traverses the tree from root to leaf by following
        the splitting conditions at each internal node.

        Args:
            x: Single sample feature vector

        Returns:
            int: Predicted class label
        """
        node = self.root

        # Traverse tree until reaching a leaf node
        while not node.is_leaf():
            # Follow left branch if feature value <= threshold
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        # Return the prediction from the leaf node
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple samples.

        Args:
            X: Feature matrix for prediction (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        # Check if model has been trained
        if self.root is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy array if needed
        X = np.array(X)

        # Predict each sample individually
        predictions = []
        for i in range(len(X)):
            prediction = self._predict_sample(X[i])
            predictions.append(prediction)

        return np.array(predictions)

    def get_depth(self) -> int:
        """
        Calculate the actual depth of the trained tree.

        Returns:
            int: Maximum depth of the tree
        """
        if self.root is None:
            return 0

        def _calculate_depth(node: Node) -> int:
            """Recursively calculate tree depth."""
            if node.is_leaf():
                return 1

            left_depth = _calculate_depth(node.left) if node.left else 0
            right_depth = _calculate_depth(node.right) if node.right else 0

            return 1 + max(left_depth, right_depth)

        return _calculate_depth(self.root)

    def get_n_leaves(self) -> int:
        """
        Count the number of leaf nodes in the tree.

        Returns:
            int: Number of leaf nodes
        """
        if self.root is None:
            return 0

        def _count_leaves(node: Node) -> int:
            """Recursively count leaf nodes."""
            if node.is_leaf():
                return 1

            left_leaves = _count_leaves(node.left) if node.left else 0
            right_leaves = _count_leaves(node.right) if node.right else 0

            return left_leaves + right_leaves

        return _count_leaves(self.root)

    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        """
        Print a text representation of the decision tree.

        This method provides a human-readable view of the tree structure
        showing the splitting conditions and leaf predictions.

        Args:
            node: Current node to print (starts with root if None)
            depth: Current depth for indentation
        """
        if node is None:
            node = self.root

        if node is None:
            print("Tree is empty (not fitted)")
            return

        # Create indentation based on depth
        indent = "  " * depth

        if node.is_leaf():
            # Print leaf node with prediction
            print(f"{indent}Leaf: class={node.prediction}, samples={node.samples}, gini={node.gini:.3f}")
        else:
            # Print internal node with splitting condition
            print(
                f"{indent}Node: feature_{node.feature} <= {node.threshold:.3f}, samples={node.samples}, gini={node.gini:.3f}")

            # Recursively print children
            if node.left:
                print(f"{indent}├─ True:")
                self.print_tree(node.left, depth + 1)
            if node.right:
                print(f"{indent}└─ False:")
                self.print_tree(node.right, depth + 1)