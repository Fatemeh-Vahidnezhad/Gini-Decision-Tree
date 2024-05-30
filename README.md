# Gini Impurity Calculator

## Overview
This repository contains a Python implementation of the Gini impurity calculator, which is used to evaluate
the quality of splits in a dataset.
The Gini file is fundamental in building decision tree models for classification tasks. The current implementation
allows for computing the best split based on Gini impurity.

## Features
- Calculation of Gini impurity for given groups in a dataset.
- Identification of the best-split point in the dataset based on the lowest Gini impurity.
- Scalability to handle datasets with multiple features.

## Future Work
This project is a work in progress, and future enhancements will include:
- Full decision tree algorithm implementation.

## Installation
Clone this repository to your local machine

## Usage
To use this Gini impurity calculator, follow these steps:
1. Prepare your dataset in a list of lists format, where each inner list represents a data record, and the last element of each list is the class label.
2. Import the `ComputeGini` class from the module.
3. Create an instance of `ComputeGini` with your dataset.
4. Call the `best_gini()` method to find the best split.

Example:
```python
from compute_gini import ComputeGini

dataset = [[1, 6, 1], [2, 5, 0], [3, 8, 1], [4, 4, 0]]
gini_calculator = ComputeGini(dataset)
best_group, best_gini, best_value, best_index = gini_calculator.best_gini()
print(f'Best Group: {best_group}, Best Gini: {best_gini}, Best Value: {best_value}, Best Index: {best_index}')

