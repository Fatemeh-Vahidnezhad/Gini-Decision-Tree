import numpy as np


class Evaluation:
    
    def Mean_Square_Error(self, dataset, y_pred):
        y = [row[-1] for row in dataset]
        y = np.array(y)
        y_pred = np.array(y_pred)
        return np.mean((y-y_pred)**2)