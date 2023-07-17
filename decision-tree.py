import numpy as np

class DecisionTree:

    def __init__(self, max_depth = 6, depth = 1):
        self.max_depth = max_depth
        self.depth = depth
        self.left = None
        self.right = None

    def fit(self, data, target):
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)

    def __calculate_impurity_score(self, data):
        if data is None or data.empty: return 0
        p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist()
        return p_i * (1 - p_i) * 2
    
    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for row in data.values])
    
    def __flow_data_thru_tree(self, row):
        return self.data[self.target].value_counts() \
                    .apply(lambda x: x/len(self.data)).tolist()
    
