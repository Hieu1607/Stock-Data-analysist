import numpy as np
from collections import Counter

def prepare_data_for_decision_tree(df , feature_columns = None, target_column = None, method_to_create_threshold = None ):
    """
    Prepare data for decision tree training.
    Args:
        df (pd.DataFrame): Input dataframe.
        feature_columns (list): List of feature columns. If None, all columns except target_column are used.
        target_column (str): Name of the target column.
    Returns:
        np.ndarray: Features and labels for training.
    """
    # Check if the data is empty
    if feature_columns is None:
        X = df.drop(columns=[target_column]).values
    else:
        X = df[feature_columns].values
    if target_column is None:   
        target_column = "daily_returns"
    else:
        target_column = target_column
    if method_to_create_threshold is None:
        method_to_create_threshold = 'quantile'
    else:
        method_to_create_threshold = method_to_create_threshold

    # Create target variable
    df['future_returns'] = df[target_column].shift(-1)
    df['future_returns'].fillna(0, inplace=True)

    # Create labels based on method_to_create_threshold    
    if method_to_create_threshold == 'quantile':
        up_threshold = df["future_returns"].quantile(0.66)
        down_threshold = df["future_returns"].quantile(0.33)
        df['trend_label'] = np.where(df["future_returns"] > up_threshold, 1, np.where(df["future_returns"] < down_threshold, -1, 0))
    if method_to_create_threshold == 'specific':
        up_threshold = 0.01
        down_threshold = -0.01
        df['trend_label'] = np.where(df["future_returns"] > up_threshold, 1, np.where(df["future_returns"] < down_threshold, -1, 0))
    if method_to_create_threshold == 'up_and_down':
        df['up_threshold'] = (df['High'] - df['Open']) / df['Open']
        df['down_threshold'] = (df['Open'] - df['Low']) / df['Open']
        
        def get_trend_label(row):
            if row["future_returns"] > row["up_threshold"]:
                return 1      # Uptrend
            elif row["future_returns"] < row["down_threshold"]:
                return -1     # Downtrend
            else:
                return 0      # Sideway

        df["trend_label"] = df.apply(get_trend_label, axis=1)
    y = df['trend_label'].values    
    return X, y

class DecisionTree :
    def __init__(self,max_depth = 10,min_sample_split = 2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.tree = None
    def _gini(self,y):
        """
        Calculate the Gini impurity of a dataset.
        Args:
            y (np.ndarray): Labels of the dataset.
        Returns:
            float: Gini impurity.
        """
        
        counter = Counter(y)
        impurity = 1.0
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity

    def _best_split(self,X, y):
        """
        Find the best split for a dataset.
        Args:
            X (np.ndarray): Features of the dataset.
            y (np.ndarray): Labels of the dataset.
        Returns:
            tuple: Best feature index, best threshold, and best Gini impurity.    
        """

        best_gain = -1
        best_split = None
        n_features = X.shape[1]
        original_impurity = self._gini(y)

        for feature_index in range(n_features):
            feature_values = np.unique(X[:, feature_index])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # Calculate Gini impurity for the split
                left_impurity = self._gini(left_y)
                right_impurity = self._gini(right_y)

                # Calculate the weighted average of Gini impurity
                weighted_impurity = (len(left_y) * left_impurity + len(right_y) * right_impurity) / len(y)

                # Calculate the information gain
                gain = original_impurity  - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold)
            
        return best_split, best_gain

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        Args:
            X (np.ndarray): Features of the dataset.
            y (np.ndarray): Labels of the dataset.
            depth (int): Current depth of the tree.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Base case
        if (depth >= self.max_depth)  or n_samples < self.min_sample_split or n_classes == 1:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        # Find the best split
        split, gain = self._best_split(X, y)
        if split is None or gain == 0:
            return {'class': Counter(y).most_common(1)[0][0]}
        feature_index, threshold = split


        # Split data
        left_mask = X[:,feature_index] <= threshold
        right_mask = ~left_mask

        left_X,left_y = X[left_mask],y[left_mask]
        right_X,right_y = X[right_mask],y[right_mask]

        # Create a node
        return {
            'feature_index' : feature_index,
            'threshold' : threshold,
            'left' : self._build_tree(left_X,left_y,depth + 1),
            'right' : self._build_tree(right_X,right_y,depth + 1)
        }
    def fit(self,X,y):
        """
        Fit the decision tree to the dataset.
        Args:
            X (np.ndarray): Features of the dataset.
            y (np.ndarray): Labels of the dataset.
        """
        self.tree = self._build_tree(X,y,0)
    def _predict_one(self, x, node):
        """ Predict the class of a single sample using the decision tree."""
        if 'class' in node:
            return node['class']
        if x[node['feature_index']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    def predict(self,X):
        """
        Predict the class of samples in the dataset.
        Args:
            X (np.ndarray): Features of the dataset.
        Returns:
            np.ndarray: Predicted classes.
        """
        return np.array([self._predict_one(x, self.tree) for x in X])
    def score(self, y_true, y_pred):
        """
        Calculate the accuracy of the model.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: Accuracy of the model.
        """
        return np.mean(y_true == y_pred)