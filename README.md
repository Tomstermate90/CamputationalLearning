# Custom One-vs-All Classification Exercise

## Overview
This project implements a custom One-vs-All classifier from scratch and compares it with the Softmax approach using the CIFAR-10 dataset. The key goals are to:

- Implement a fully custom One-vs-All (OvA) classifier from scratch
- Compare performance metrics between the custom OvA and Softmax approaches
- Analyze confusion matrices to identify difficult-to-distinguish class pairs
- Create specialized binary classifiers for problematic class pairs
- Implement a hybrid approach to improve overall classification accuracy

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

If these files are not found, the notebook will automatically generate sample data for demonstration purposes.

## Custom One-vs-All Implementation
The core of this project is the `CustomOneVsAllClassifier` class that builds a multi-class classifier by:

1. Creating a separate binary classifier for each class
2. Training each classifier to distinguish one class from all others
3. Using the classifier with the highest confidence score for predictions

Key methods of the custom classifier:
- `fit(X, y)`: Trains one binary classifier per class
- `predict(X)`: Returns the class with highest confidence score for each sample
- `predict_proba(X)`: Returns probability estimates for each class

## Notebook Structure
The Jupyter notebook is organized into 13 logical cells:

1. **Import Libraries and Setup**: Initialization of dependencies
2. **Helper Functions**: Utility functions for data handling and visualization
3. **Custom One-vs-All Classifier Implementation**: The core custom OvA class
4. **Data Loading and Preparation**: Loading dataset and train-test splitting (70/30)
5. **Softmax Logistic Regression**: Training and evaluation of Softmax model
6. **Custom One-vs-All Implementation**: Training and evaluation of custom OvA
7. **Basic Metrics Comparison**: Comparing accuracy and training time
8. **Cost Function and F1-Score Analysis**: Additional performance metrics
9. **Confusion Matrix Analysis**: Identifying difficult class pairs
10. **Specialized Binary Classifier**: Creating targeted models for confused classes
11. **Hybrid Approach Implementation**: Combining OvA with specialized classifier
12. **Final Comparisons**: Comprehensive visual comparison of all approaches
13. **Conclusions and Insights**: Summary and analysis of results

## Key Functions

### `load_data(features_file, labels_file)`
Loads the feature and label data from NumPy files, with error handling for missing files.

### `testmymodel(model, x_features, y_labels)`
Tests any model (that has a `predict` method) on given features and labels, returning accuracy percentage and displaying a classification report and F1-score.

### `CustomOneVsAllClassifier`
A complete custom implementation of One-vs-All classification that:
- Creates and trains a separate binary classifier for each class
- Provides methods for prediction and probability estimation
- Can be used with any base classifier that follows scikit-learn's API

### `plot_performance_comparison(metrics_dict, title)`
Creates visual comparisons between different models' performance metrics.

## Experimental Process

1. **Data Preparation**: The CIFAR-10 dataset is split into 70% training and 30% testing sets
2. **Model Training**: 
   - Softmax model trained using scikit-learn's LogisticRegression with `multi_class='multinomial'`
   - Custom OvA model trained by creating binary classifiers for each class
3. **Performance Evaluation**: Both models are evaluated on accuracy, training time, cost function, and F1-score
4. **Confusion Matrix Analysis**: The confusion matrix for the custom OvA model is analyzed to find the most confused class pairs
5. **Specialized Classification**: A binary classifier is created specifically for the most confused classes
6. **Hybrid Approach**: A combined approach that uses the specialized classifier for difficult classes
7. **Comparative Analysis**: All approaches are compared with visualizations

## Visualizations
The notebook generates multiple visualizations to help interpret results:

1. **Bar charts** comparing accuracy, training time, log loss, and F1-score between models
2. **Confusion matrix heatmaps** showing classification patterns
3. **Specialized visualizations** highlighting misclassifications
4. **Improvement analysis charts** showing the effect of the hybrid approach

## Usage
1. Place the CIFAR-10 feature and label files in the same directory as the notebook
2. If your files have different names, modify the file paths in the `load_data` function call
3. Run the notebook cells sequentially
4. Examine the output and visualizations to understand the custom OvA implementation

## Benefits of Custom Implementation
1. **Educational Value**: Understand the inner workings of OvA classification
2. **Transparency**: See how confidence scores are handled between classifiers
3. **Customization**: Easily modify individual binary classifiers or the aggregation logic
4. **Specialized Handling**: Implement custom logic for specific class pairs

## Notes
- The notebook includes proper error handling for missing files
- The custom OvA implementation may be slower than scikit-learn's optimized version
- The hybrid approach demonstrates an important concept: using specialized classifiers for difficult classification problems