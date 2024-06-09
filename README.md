## Decision Tree and Gini Calculator
This repository contains two primary components: a Gini Impurity Calculator and a Decision Tree Classifier. 
These tools are designed to aid in understanding and implementing machine learning classification algorithms, particularly those involving decision trees.

## Gini Calculator
This repository contains a Python implementation of the Gini impurity calculator, which is used to evaluate
the quality of splits in a dataset.
The Gini file is fundamental in building decision tree models for classification tasks. The current implementation
allows for computing the best split based on Gini impurity.

## Decision Tree Classifier
The Decision Tree Classifier is an implementation of a basic decision tree that uses the Gini impurity calculator to find the best split among the features.
The tree recursively splits the data into subsets that gain the highest purity.

Features:
    Build a decision tree based on Gini impurity.
    Predict labels for new data using the trained tree.
    Handle depth control and minimum size for splitting.

## Features
- Calculation of Gini impurity for given groups in a dataset.
- Identification of the best-split point in the dataset based on the lowest Gini impurity.
- Scalability to handle datasets with multiple features.


## Installation
Clone this repository to your local machine

## Usage
To use this Gini impurity calculator, follow these steps:
1. Prepare your dataset in a list of lists format, where each inner list represents a data record, and the last element of each list is the class label.
2. Import the `ComputeGini` class from the module.
3. Create an instance of `ComputeGini` with your dataset.
4. Call the `best_gini()` method to find the best split.




