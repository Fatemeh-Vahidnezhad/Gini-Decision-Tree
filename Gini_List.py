class ComputeGini:
    def __init__(self, data):
        self.dic = {}
        self.data = data
        self.len_data = len(self.data)
        self.count_features = len(self.data[0])-1 if self.data else 0

    def gini_entropy(self, group, index_col):
        # gini = 1-sum(pi**2) , pi = count(label i)/ len(group)
        if not group:
            return 0
        impurity = 1
        len_group = len(group)
        label_count = {row[-1] for row in group}
        dic = {i: 0 for i in label_count}
        for key in dic.keys():
            dic[key] = [row[-1] for row in group].count(key)
        for val in dic.values():
            impurity -= (val/len_group)**2
        return impurity

    def grouping(self, value, index_col):
        left = []
        right = []
        for row in self.data:
            if row[index_col] <= value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def best_gini(self):
        best_gini = float('inf')
        best_group = []
        best_value = None
        best_index = -1
        for index_col in range(self.count_features):
            for row in self.data:
                groups = self.grouping(row[index_col], index_col)
                gini = 0
                for group in groups:
                    gini_group = self.gini_entropy(group, index_col)
                    gini += (gini_group * (len(group)/self.len_data))
                if gini < best_gini:
                    best_value = row[index_col]
                    best_gini = gini
                    best_group = groups
                    best_index = index_col
        return best_group, best_gini, best_value, best_index


dataset = [[1, 6, 1],
           [2, 5, 0],
           [3, 8, 1],
           [4, 4, 0]]
# dataset = [[1, 1],
#            [2, 1],
#            [3, 1],
#            [4, 0]]
gini1 = ComputeGini(dataset)
best_group, best_gini, best_value, best_index = gini1.best_gini()
print(f'best_group: {best_group}\n, best_gini: {best_gini}\n, best_value: {best_value}\n, best_index: {best_index}\n')
