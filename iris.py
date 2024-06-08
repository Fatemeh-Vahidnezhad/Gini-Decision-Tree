from sklearn.datasets import load_iris
from decision_tree import *
import numpy as np
from evaluation import *


# load dataset
iris = load_iris()
x = iris.data
y = iris.target
y_reshaped = y.reshape(-1,1)
dataset = np.concatenate([x, y_reshaped], axis=1)



tree = make_tree(dataset)
y_pred = decision_tree_prediction(dataset, tree)
eval = Evaluation()
print('Mean_Square_Error:', eval.Mean_Square_Error(dataset, y_pred))