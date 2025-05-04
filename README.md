# One-vs-All Classification Exercise with Advanced Visualization

## Overview
This project implements and compares two logistic regression approaches for multi-class classification using the CIFAR-10 dataset:
1. Softmax (multinomial) logistic regression
2. One-vs-All (OvR) logistic regression

The main goals are to:
- Compare the performance (accuracy, F1-score) of both methods
- Compare the computational efficiency (training time) of both methods
- Calculate the cost function (log loss) for each approach
- Identify difficult-to-distinguish class pairs using confusion matrices
- Implement a specialized binary classifier for the most confused classes and evaluate if it improves performance
- Analyze and visualize improvements through comprehensive comparative graphics

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required packages using:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset
The exercise uses features extracted from the CIFAR-10 dataset. The original dataset can be viewed at: https://www.cs.toronto.edu/~kriz/cifar.html

The data files required:
- `cifar10_features.npy`: A NumPy array file containing 16 extracted features per image
- `cifar10_labels.npy`: A NumPy array file containing class labels (0-9)

## Notebook Structure
The Jupyter notebook is organized into the following logical sections:

1. **Import Libraries and Setup**: Initialization of dependencies and environment
2. **Helper Functions**: Core utility functions for data loading, model evaluation, and visualization
3. **Data Loading and Preparation**: Loading dataset and train-test splitting
4. **Softmax Logistic Regression**: Training and evaluation of multinomial model
5. **One-vs-All Logistic Regression**: Training and evaluation of OvR model
6. **Accuracy and Runtime Comparison**: Visual comparison of performance metrics
7. **Cost Function and F1-Score Analysis**: Calculation and visualization of additional metrics
8. **Confusion Matrix Analysis**: Detailed analysis of classification errors
9. **Specialized Binary Classifier**: Targeted model for most confused classes
10. **Hybrid Approach Implementation**: Combined model leveraging specialized classifier
11. **Final Comparisons**: Comprehensive visual comparison of all approaches
12. **Conclusions and Findings**: Summary of results and insights

## Key Functions

### `load_data(features_file, labels_file)`
Loads the feature and label data from NumPy files.

### `testmymodel(model, x_features, y_labels)`
Tests a trained model on given features and labels, returning the accuracy percentage. This function also prints:
- Model accuracy
- Classification report
- F1-mean score

### `plot_performance_comparison(metrics_dict, title)`
Creates bar charts comparing performance metrics between different models:
- Accepts a dictionary mapping model names to metric values
- Produces a labeled bar chart with value annotations
- Used for visualizing accuracy, training time, log loss, and F1-score comparisons

## Experimental Process
1. The dataset is split into 70% training and 30% testing sets
2. Two logistic regression models are trained:
   - Softmax (multinomial) using 'lbfgs' solver
   - One-vs-All (OvR) using 'liblinear' solver
3. Performance metrics are calculated and visualized for both models
4. The confusion matrix for the One-vs-All model is analyzed to find the most confused class pairs
5. A specialized binary classifier is created for the most confused classes
6. A hybrid approach is implemented and evaluated
7. Comprehensive visualizations are generated to compare all approaches

## Visualizations
The notebook generates multiple visualizations to help interpret the results:

1. **Performance Comparisons**:
   - Bar charts for accuracy, training time, log loss, and F1-score
   - Direct visual comparison between all methods (Softmax, OvA, Hybrid)

2. **Confusion Matrix Analysis**:
   - Standard confusion matrix heatmap
   - Specialized heatmap highlighting only misclassifications (off-diagonal values)
   - Targeted visualization of most confused class pairs

3. **Improvement Analysis**:
   - Bar chart showing before/after misclassifications for confused classes
   - Visual representation of confusion reduction in the hybrid approach

## Expected Output
The notebook will generate:
1. Training times and accuracy scores with visualizations
2. Cost function values (log loss) with comparative charts
3. F1-mean scores with visual comparison
4. Multiple confusion matrix visualizations
5. Identification of the most confused class pairs
6. Performance metrics for the specialized binary classifier
7. Comprehensive comparison of the hybrid approach with the original methods
8. Detailed insights about confusion reduction and model improvements

## Usage
1. Place your feature and label files in the same directory as the notebook
2. If your files have different names than 'cifar10_features.npy' and 'cifar10_labels.npy', modify the file paths in the `load_data` function call
3. Run the notebook cells sequentially to see progressive results and visualizations
4. Experiment with different random states or model parameters by modifying the relevant code cells

## Notes
- The notebook provides detailed documentation at each step to explain the process
- Error handling is included for missing files, with sample data generation as a fallback
- The hybrid approach demonstrates how targeted binary classifiers can improve multi-class classification
- The visualization tools can be repurposed for other classification problems
- Additional insights section provides directions for further improvements and experimentation

## Extending the Project
Some ways to extend this project:
1. Apply the same approach to different datasets
2. Try different classification algorithms instead of logistic regression
3. Implement feature engineering to improve classification accuracy
4. Add more specialized binary classifiers for multiple confused class pairs
5. Implement hyperparameter tuning for the models
6. Create an interactive dashboard for model comparison