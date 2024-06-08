from Gini_List import *


class DicisionTree:
    def __init__(self, best_threshold, left, right, best_colindex, label=None) -> None:
        self.best_threshold = best_threshold
        self.right = right
        self.left = left
        self.best_colindex = best_colindex
        self.label = label


def common_label(dataset):
    if dataset:
        return max(set(row[-1] for row in dataset), key=lambda x:[row[-1] for row in dataset].count(x))


def make_tree(dataset, depth=0, min_size=1):
    if len(dataset)<= min_size or depth >=3:
        return DicisionTree(best_threshold=None, left=None, right=None, best_colindex=None, label=common_label(dataset))

    gini = ComputeGini(dataset)
    groups,_, best_threshold,best_colindex = gini.best_gini()
    left, right = groups

    if best_threshold == None:
        return DicisionTree(label=common_label(dataset))

    if left == None or right == None:
        return DicisionTree(label=common_label(dataset))
    
    left_child = make_tree(left, depth+1)

    right_child = make_tree(right, depth+1)
    return DicisionTree(best_threshold=best_threshold, left=left_child, right=right_child, best_colindex=best_colindex)


def prediction(node, row):
    if node.label is not None:
        return node.label
    if row[node.best_colindex] <= node.best_threshold:
        return prediction(node.left, row)
    else:
        return prediction(node.right, row)
    

def decision_tree_prediction(dataset, tree):
    return [prediction(tree, row) for row in dataset]


