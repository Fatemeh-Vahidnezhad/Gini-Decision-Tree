from sklearn.datasets import load_iris
from decision_tree import *
import numpy as np
from evaluation import *
import pandas as pd
from sklearn.utils import resample
import time

# load dataset
iris = load_iris()
x = iris.data
y = iris.target
y_reshaped = y.reshape(-1,1)
dataset = np.concatenate([x, y_reshaped], axis=1)




# df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df_target = pd.DataFrame(data=iris.target, columns=['target'])
# Resample data
# resampled_data = resample(df_iris, replace=True, n_samples=50000, random_state=123)
# resampled_data_target = resample(df_target, replace=True, n_samples=50000, random_state=123)

# df = pd.concat([resampled_data, resampled_data_target], axis=1)
# print(df.shape)

# time1 = time.time()
# print('time1', time1)
tree = make_tree(dataset)
y_pred = decision_tree_prediction(dataset, tree)
eval = Evaluation()
print('Mean_Square_Error:', eval.Mean_Square_Error(dataset, y_pred))
# time2 = time.time()
# print(time2 - time1)
