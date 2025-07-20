"""
AdaBoost Classifier Implementation from Scratch
===============================================

This module implements the AdaBoost (Adaptive Boosting) algorithm from scratch.
AdaBoost is a boosting ensemble method that combines multiple weak learners
(typically decision stumps) into a strong classifier.

Key Concepts:
1. Sequential Learning: Each weak learner is trained on a weighted version of data
2. Sample Weighting: Misclassified samples get higher weights for next iteration
3. Weak Learner Weighting: Better performing weak learners get higher voting weights
4. Final Prediction: Weighted majority vote of all weak learners

Mathematical Foundation:
- Sample weight update: w_i = w_i * exp(α * I(h(x_i) ≠ y_i))
- Weak learner weight: α = 0.5 * ln((1 - error) / error)
- Final prediction: sign(Σ α_t * h_t(x))

AdaBoost Properties:
- Focuses on hard-to-classify examples
- Can reduce both bias and variance
- Sensitive to noise and outliers
- Tends to converge to low training error

"""

import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier


class DecisionStump:
    """
    Decision Stump: A decision tree with maximum depth of 1.

    A decision stump is the simplest possible decision tree with just one split.
    It's the default weak learner for AdaBoost because:
    1. It's simple and fast to train
    2. It's guaranteed to be a weak learner (better than random guessing)
    3. Multiple stumps can capture complex decision boundaries

    The stump makes decisions based on a single feature and threshold:
    - If feature_value <= threshold: predict class_left
    - If feature_value > threshold: predict class_right
    """

    def __init__(self):
        """Initialize an empty decision stump."""
        self.feature_idx = None  # Index of the feature to split on
        self.threshold = None  # Threshold value for splitting
        self.class_left = None  # Prediction for left branch (<=)
        self.class_right = None  # Prediction for right branch (>)
        self.error = float('inf')  # Training error of this stump

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> None:
        """
        Train the decision stump on weighted data.

        The training process finds the best single feature and threshold
        that minimizes the weighted error rate. For each possible split:
        1. Calculate weighted error if this split is used
        2. Keep track of the split with minimum weighted error

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            sample_weights: Sample weights (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize best parameters
        best_error = float('inf')
        best_feature = None
        best_threshold = None
        best_class_left = None
        best_class_right = None

        # Try each feature as a potential split
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]

            # Try each unique value as a threshold
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # Split data based on threshold
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Skip if one side is empty
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Get labels for each side
                y_left = y[left_mask]
                y_right = y[right_mask]

                # Get weights for each side
                weights_left = sample_weights[left_mask]
                weights_right = sample_weights[right_mask]

                # Determine majority class for each side (weighted)
                classes = np.unique(y)

                # For left side
                best_left_class = None
                best_left_error = float('inf')
                for cls in classes:
                    # Calculate weighted error if we predict this class for left side
                    error = np.sum(weights_left[y_left != cls])
                    if error < best_left_error:
                        best_left_error = error
                        best_left_class = cls

                # For right side
                best_right_class = None
                best_right_error = float('inf')
                for cls in classes:
                    # Calculate weighted error if we predict this class for right side
                    error = np.sum(weights_right[y_right != cls])
                    if error < best_right_error:
                        best_right_error = error
                        best_right_class = cls

                # Total weighted error for this split
                total_error = best_left_error + best_right_error

                # Update best split if this is better
                if total_error < best_error:
                    best_error = total_error
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_class_left = best_left_class
                    best_class_right = best_right_class

        # Store the best split found
        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.class_left = best_class_left
        self.class_right = best_class_right
        self.error = best_error

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained stump.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels
        """
        if self.feature_idx is None:
            raise ValueError("Stump must be fitted before making predictions")

        predictions = np.zeros(len(X))

        # Apply the learned split
        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask

        predictions[left_mask] = self.class_left
        predictions[right_mask] = self.class_right

        return predictions.astype(int)


class AdaBoostClassifier:
    """
    AdaBoost Classifier implemented from scratch.

    AdaBoost (Adaptive Boosting) works by:
    1. Training weak learners sequentially on weighted data
    2. Increasing weights of misclassified samples after each iteration
    3. Computing importance weights for each weak learner based on performance
    4. Making final predictions using weighted majority vote

    The algorithm adaptively focuses on hard-to-classify examples,
    potentially achieving very low training error even with weak learners.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0,
                 algorithm: str = 'SAMME', random_state: Optional[int] = None):
        """
        Initialize AdaBoost classifier.

        Args:
            n_estimators: Maximum number of weak learners to train
            learning_rate: Learning rate shrinks contribution of each classifier
            algorithm: Boosting algorithm ('SAMME' for discrete AdaBoost)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators  # Number of weak learners
        self.learning_rate = learning_rate  # Shrinkage parameter
        self.algorithm = algorithm  # Boosting algorithm type
        self.random_state = random_state  # Random seed

        # Initialize containers for trained components
        self.estimators_ = []  # List of weak learners
        self.estimator_weights_ = []  # Weight of each weak learner
        self.estimator_errors_ = []  # Error of each weak learner
        self.classes_ = None  # Unique class labels
        self.n_classes_ = None  # Number of classes
        self.feature_importances_ = None  # Feature importance scores

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

    def _boost(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray,
               estimator_idx: int) -> Tuple[float, float]:
        """
        Perform one boosting iteration.

        This method implements the core AdaBoost algorithm for one iteration:
        1. Train weak learner on weighted samples
        2. Calculate weighted error rate
        3. Compute weak learner weight (alpha)
        4. Update sample weights based on predictions

        Mathematical details:
        - Weighted error: ε = Σ w_i * I(h(x_i) ≠ y_i) / Σ w_i
        - Weak learner weight: α = ln((1-ε)/ε) + ln(K-1) where K is number of classes
        - Sample weight update: w_i = w_i * exp(α * I(h(x_i) ≠ y_i))

        Args:
            X: Feature matrix
            y: Target labels
            sample_weights: Current sample weights
            estimator_idx: Index of current estimator

        Returns:
            Tuple of (estimator_weight, estimator_error)
        """
        # Create and train weak learner (decision stump)
        estimator = DecisionStump()
        estimator.fit(X, y, sample_weights)

        # Make predictions
        y_pred = estimator.predict(X)

        # Calculate weighted error rate
        # Error is the sum of weights of misclassified samples
        incorrect_predictions = (y_pred != y)
        estimator_error = np.sum(sample_weights * incorrect_predictions) / np.sum(sample_weights)

        # Handle edge cases
        if estimator_error == 0:
            # Perfect classifier - give it maximum weight
            estimator_weight = 1.0
            # Don't update sample weights (they're already optimal)
            return estimator_weight, estimator_error

        if estimator_error >= 1 - (1.0 / self.n_classes_):
            # Classifier is worse than random guessing
            # In practice, this shouldn't happen with proper weak learners
            if len(self.estimators_) == 0:
                raise ValueError("First weak learner failed. Check your data.")
            else:
                # Stop boosting - no improvement possible
                return None, estimator_error

        # Calculate weak learner weight (alpha)
        # For multi-class: α = ln((1-ε)/ε) + ln(K-1)
        estimator_weight = self.learning_rate * (
                np.log((1 - estimator_error) / estimator_error) +
                np.log(self.n_classes_ - 1)
        )

        # Update sample weights
        # Increase weights of misclassified samples
        sample_weights *= np.exp(estimator_weight * incorrect_predictions)

        # Normalize weights to prevent numerical overflow
        sample_weights /= np.sum(sample_weights)

        # Store the trained estimator
        self.estimators_.append(estimator)
        self.estimator_weights_.append(estimator_weight)
        self.estimator_errors_.append(estimator_error)

        return estimator_weight, estimator_error

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifier':
        """
        Train the AdaBoost classifier.

        The training process:
        1. Initialize uniform sample weights
        2. For each boosting iteration:
           a. Train weak learner on weighted samples
           b. Calculate learner weight based on performance
           c. Update sample weights (increase for misclassified)
        3. Store all weak learners and their weights

        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Training target labels (n_samples,)

        Returns:
            self: The fitted AdaBoost classifier
        """
        # Input validation
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) == 0:
            raise ValueError("Cannot fit with empty dataset")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Store dataset information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Check for binary classification requirement
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes for classification")

        # Initialize sample weights uniformly
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples

        # Clear previous training results
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        print(f"Training AdaBoost with {self.n_estimators} estimators...")

        # Boosting iterations
        for estimator_idx in range(self.n_estimators):
            # Progress indicator
            if (estimator_idx + 1) % 10 == 0:
                print(f"Training estimator {estimator_idx + 1}/{self.n_estimators}")

            # Perform one boosting iteration
            estimator_weight, estimator_error = self._boost(
                X, y, sample_weights, estimator_idx
            )

            # Check if boosting should stop
            if estimator_weight is None:
                print(f"Early stopping at iteration {estimator_idx + 1}")
                break

            # Stop if we achieve perfect classification
            if estimator_error == 0:
                print(f"Perfect classification achieved at iteration {estimator_idx + 1}")
                break

        # Calculate feature importance
        self._calculate_feature_importance(X.shape[1])

        print(f"AdaBoost training completed with {len(self.estimators_)} estimators!")
        return self

    def _calculate_feature_importance(self, n_features: int) -> None:
        """
        Calculate feature importance based on weak learner usage.

        Feature importance in AdaBoost is calculated as the sum of weights
        of all weak learners that use each feature, weighted by their
        performance (estimator weights).

        Args:
            n_features: Number of features in the dataset
        """
        self.feature_importances_ = np.zeros(n_features)

        # Sum weighted usage of each feature
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            if hasattr(estimator, 'feature_idx') and estimator.feature_idx is not None:
                self.feature_importances_[estimator.feature_idx] += weight

        # Normalize to sum to 1
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for samples.

        The decision function returns the weighted sum of weak learner
        predictions for each class. This provides a confidence measure
        for the final predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Decision function values (n_samples, n_classes)
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before computing decision function")

        X = np.array(X)
        n_samples = len(X)

        # Initialize decision matrix
        decision = np.zeros((n_samples, self.n_classes_))

        # Sum weighted predictions from all estimators
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            # Get predictions from current estimator
            predictions = estimator.predict(X)

            # Convert to one-hot encoding and weight
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                decision[i, class_idx] += weight

        return decision

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using weighted majority vote.

        The final prediction is the class with the highest weighted vote
        from all weak learners.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels
        """
        # Get decision function values
        decision = self.decision_function(X)

        # Return class with highest decision value
        class_indices = np.argmax(decision, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using softmax of decision function.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes)
        """
        decision = self.decision_function(X)

        # Apply softmax to convert to probabilities
        exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        probabilities = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on given data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def staged_predict(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Return staged predictions (after each boosting iteration).

        This is useful for analyzing the convergence of the algorithm
        and choosing the optimal number of estimators.

        Args:
            X: Feature matrix

        Returns:
            List[np.ndarray]: Predictions after each iteration
        """
        X = np.array(X)
        n_samples = len(X)
        staged_predictions = []

        # Cumulative decision function
        decision = np.zeros((n_samples, self.n_classes_))

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            # Add current estimator's contribution
            predictions = estimator.predict(X)

            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                decision[i, class_idx] += weight

            # Get current staged prediction
            class_indices = np.argmax(decision, axis=1)
            current_predictions = self.classes_[class_indices]
            staged_predictions.append(current_predictions.copy())

        return staged_predictions

    def staged_score(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Return staged accuracy scores (after each boosting iteration).

        This helps analyze how the accuracy improves with more estimators
        and can help detect overfitting.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            List[float]: Accuracy scores after each iteration
        """
        staged_predictions = self.staged_predict(X)
        staged_scores = []

        for predictions in staged_predictions:
            accuracy = np.mean(predictions == y)
            staged_scores.append(accuracy)

        return staged_scores

    def plot_learning_curve(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        """
        Plot learning curves showing training and test accuracy vs iterations.

        This visualization helps understand:
        - How quickly the algorithm converges
        - Whether overfitting occurs
        - Optimal number of estimators

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
        """
        # Get staged scores for training data
        train_scores = self.staged_score(X_train, y_train)
        iterations = range(1, len(train_scores) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_scores, 'b-', label='Training Accuracy', linewidth=2)

        # Plot test scores if provided
        if X_test is not None and y_test is not None:
            test_scores = self.staged_score(X_test, y_test)
            plt.plot(iterations, test_scores, 'r-', label='Test Accuracy', linewidth=2)

        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title('AdaBoost Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names: List[str] = None, top_n: int = 20) -> None:
        """
        Plot feature importance scores.

        Args:
            feature_names: List of feature names
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
        plt.title(f'AdaBoost Feature Importance (Top {len(names)})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_estimator_weights(self) -> None:
        """
        Plot the weights of individual estimators.

        This shows how much each weak learner contributes to the final prediction.
        Higher weights indicate better performing estimators.
        """
        if not self.estimator_weights_:
            print("No estimator weights available. Model may not be fitted.")
            return

        plt.figure(figsize=(12, 6))

        # Plot estimator weights
        iterations = range(1, len(self.estimator_weights_) + 1)
        plt.subplot(1, 2, 1)
        plt.bar(iterations, self.estimator_weights_)
        plt.xlabel('Estimator Number')
        plt.ylabel('Estimator Weight (Alpha)')
        plt.title('AdaBoost Estimator Weights')
        plt.grid(True, alpha=0.3)

        # Plot estimator errors
        plt.subplot(1, 2, 2)
        plt.bar(iterations, self.estimator_errors_)
        plt.xlabel('Estimator Number')
        plt.ylabel('Estimator Error')
        plt.title('AdaBoost Estimator Errors')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_estimator_info(self) -> List[dict]:
        """
        Get detailed information about each estimator.

        Returns:
            List[dict]: Information about each estimator including
                       feature used, threshold, weight, and error
        """
        estimator_info = []

        for i, (estimator, weight, error) in enumerate(
                zip(self.estimators_, self.estimator_weights_, self.estimator_errors_)
        ):
            info = {
                'estimator_idx': i,
                'feature_idx': getattr(estimator, 'feature_idx', None),
                'threshold': getattr(estimator, 'threshold', None),
                'class_left': getattr(estimator, 'class_left', None),
                'class_right': getattr(estimator, 'class_right', None),
                'weight': weight,
                'error': error
            }
            estimator_info.append(info)

        return estimator_info

    def print_boost_stats(self) -> None:
        """
        Print summary statistics about the trained AdaBoost classifier.
        """
        if not self.estimators_:
            print("AdaBoost not trained yet.")
            return

        print(f"AdaBoost Statistics:")
        print(f"Number of estimators: {len(self.estimators_)}")
        print(f"Number of classes: {self.n_classes_}")
        print(f"Classes: {self.classes_}")

        if self.estimator_weights_:
            weights = np.array(self.estimator_weights_)
            errors = np.array(self.estimator_errors_)

            print(f"Estimator weights - Mean: {np.mean(weights):.3f}, Std: {np.std(weights):.3f}")
            print(f"Estimator weights - Range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
            print(f"Estimator errors - Mean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}")
            print(f"Estimator errors - Range: [{np.min(errors):.3f}, {np.max(errors):.3f}]")

        if self.feature_importances_ is not None:
            top_feature_idx = np.argmax(self.feature_importances_)
            top_feature_importance = self.feature_importances_[top_feature_idx]
            print(f"Most important feature: feature_{top_feature_idx} (importance: {top_feature_importance:.3f})")


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating AdaBoost usage.
    This section shows how to use the AdaBoost classifier
    and is useful for testing the implementation.
    """

    # Create synthetic dataset for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 5

    # Create linearly separable data with some noise
    X = np.random.randn(n_samples, n_features)

    # Create decision boundary: linear combination of first two features
    decision_boundary = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    y = (decision_boundary > 0).astype(int)

    # Add some noise to make it more challenging
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]

    # Split into train/test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train AdaBoost
    print("Testing AdaBoost Implementation...")
    ada = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    # Fit and evaluate
    ada.fit(X_train, y_train)

    # Make predictions
    train_pred = ada.predict(X_train)
    test_pred = ada.predict(X_test)

    # Calculate accuracies
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    # Print AdaBoost statistics
    ada.print_boost_stats()

    # Show staged scores to demonstrate convergence
    train_staged_scores = ada.staged_score(X_train, y_train)
    test_staged_scores = ada.staged_score(X_test, y_test)

    print(f"Final training accuracy after {len(train_staged_scores)} iterations: {train_staged_scores[-1]:.3f}")
    print(f"Final test accuracy after {len(test_staged_scores)} iterations: {test_staged_scores[-1]:.3f}")

    print("AdaBoost implementation test completed!")