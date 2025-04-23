import numpy as np
import sys
import os
from scipy.stats import mode

# Get the absolute path of the `src` folder
src_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
# Add `src` to the system path
sys.path.insert(0, src_path)
# Import 
from models.DecisionTree import DecisionTree

class RandomForest:
    def __init__(self,max_depth = 10,min_sample_split = 2):
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.trees = []
    def fit(self, Xs, ys):
        for i in range(len(Xs)):
            tree = DecisionTree(max_depth = self.max_depth,min_sample_split = self.min_samples_split)
            tree.fit(Xs[i], ys[i])
            self.trees.append(tree)
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return mode(predictions, axis=0)[0].flatten()
    def score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
