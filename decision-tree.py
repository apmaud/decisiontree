import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self, max_depth = 6, depth = 1):
        self.max_depth = max_depth
        self.depth = depth
        self.left = None
        self.right = None

    def fit(self, data, target):
        if self.depth <= self.max_depth: print(f"currently at Depth: {self.depth}")
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)
        if self.depth <= self.max_depth:
            self.__validate_data()
            self.impurity_score = self.__calculate_impurity_score(self.data[self.target])
            self.criteria, self.split_feature, self.information_gain = self.__find_best_split()
            if self.criteria is not None and self.information_gain > 0: self.__create_branches()
        else:
            print("Max Depth Reached")
    
    def __create_branches(self):
        self.left = DecisionTree(max_depth = self.max_depth, depth = self.depth + 1)
        self.right = DecisionTree(max_depth = self.max_depth, depth = self.depth + 1)
        left_rows = self.data[self.data[self.split_feature] <= self.criteria]
        right_rows = self.date[self.data[self.split_feature] > self.critera]
        self.left.fit(data = left_rows, target = self.target)
        self.right.fit(data = right_rows, target = self.target)

    def __calculate_impurity_score(self, data):
        if data is None or data.empty: return 0
        p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist()
        return p_i * (1 - p_i) * 2
    
    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for row in data.values])
    
    def __flow_data_thru_tree(self, row):
        return self.data[self.target].value_counts() \
                    .apply(lambda x: x/len(self.data)).tolist()
    
