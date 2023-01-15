import pandas as pd
from typing import List
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def to_array(values) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.values
    return values

class Imputation:
    models: List["ManyToOneImputation"]
    X_columns: List[str]
    Y_columns: List[str]

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        Train a random forest model to predict the output variables from the input variables.
        
        Args:
            X (pd.DataFrame): The dataset containing the input variables.
            Y (pd.DataFrame): The dataset containing the output variables.
        """

        self.X_columns = X.columns
        self.Y_columns = Y.columns

        X = to_array(X)
        Y = to_array(Y)

        self.models = []
        # We train a separate model for each output variable. For example, if X = [income, age] and Y = [height, weight], we train two models:
        # 1. Predict height from income and age.
        # 2. Predict weight from income, age and (predicted) height.
    
        for i in range(Y.shape[1]):
            X_ = np.concatenate([X, Y[:, :i]], axis=1)
            y_ = Y[:, i]
            model = ManyToOneImputation()
            model.train(X_, y_)
            self.models.append(model)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the output variables for the input dataset.
        
        Args:
            X (pd.DataFrame): The dataset to predict on.
        
        Returns:
            pd.DataFrame: The predicted dataset.
        """
        
        X = to_array(X)
        Y = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            X_ = np.concatenate([X, Y[:, :i]], axis=1)
            Y[:, i] = model.predict(X_)
        return pd.DataFrame(Y, columns=self.Y_columns)


class ManyToOneImputation:
    model: RandomForestRegressor

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series = None):
        """
        Train a random forest model to predict the output variable from the input variables.
        
        Args:
            X (pd.DataFrame): The dataset containing the input variables.
            y (pd.Series): The dataset containing the output variable.
            sample_weight (pd.Series): The sample weights.
        """
        
        X = to_array(X)
        y = to_array(y)
        self.model = RandomForestRegressor()
        self.model.fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the output variable for the input dataset.
        
        Args:
            X (pd.DataFrame): The dataset to predict on.
        
        Returns:
            pd.Series: The predicted dataset.
        """
        
        X = to_array(X)
        return pd.Series(self.model.predict(X))
