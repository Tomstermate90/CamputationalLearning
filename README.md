# One-vs-All Classification Exercise

## Overview
This project implements and compares two logistic regression approaches for multi-class classification using the CIFAR dataset:
1. Softmax (multinomial) logistic regression
2. One-vs-All (OvR) logistic regression

The main goals are to:
- Compare the performance (accuracy, F1-score) of both methods
- Compare the computational efficiency (training time) of both methods
- Calculate the cost function (log loss) for each approach
- Identify difficult-to-distinguish class pairs using confusion matrices
- Implement a specialized binary classifier for the most confused classes and evaluate if it improves performance

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
- A NumPy array file containing 16 extracted features per image
- A NumPy array file containing class labels (0-9)

## Usage
1. Place your feature and label files in the same directory as the notebook
2. If your files have different names than 'features.npy' and 'labels.npy', modify the file paths in the `load_data` function call
3. Run the notebook cells sequentially

## Key Functions

### `load_data(features_file, labels_file)`
Loads the feature and label data from NumPy files.

### `testmymodel(model, X_features, y_labels)`
Tests a trained model on given features and labels, returning the accuracy percentage. This function also prints:
- Model accuracy
- Classification report
- F1-mean score

## Experimental Process
1. The dataset is split into 70% training and 30% testing sets
2. Two logistic regression models are trained:
   - Softmax (multinomial) using 'lbfgs' solver
   - One-vs-All (OvR) using 'liblinear' solver
3. Performance metrics are calculated for both models
4. The confusion matrix for the One-vs-All model is analyzed to find the most confused class pairs
5. A specialized binary classifier is created for the most confused classes
6. A hybrid approach is implemented and evaluated

## Expected Output
The notebook will generate:
1. Training times and accuracy scores for both methods
2. Cost function values (log loss) for both methods
3. F1-mean scores for both methods
4. Confusion matrix visualizations
5. Identification of the most confused class pairs
6. Performance metrics for the specialized binary classifier
7. Comparison of the hybrid approach with the original One-vs-All approach

## Notes
- The `testmymodel` function can be used to evaluate models on unseen data
- The code includes error handling for missing files and will generate sample data for demonstration if needed
- The visualization elements require a graphical backend in your Jupyter environment
