import pandas as pd
from typing import List, Dict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from pathlib import Path


def to_array(values) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.values
    return values


def get_category_mapping(values: pd.Series) -> Dict[str, int]:
    return {category: i for i, category in enumerate(values.unique())}


class Imputation:
    """
    An `Imputation` represents a learned function f(`input_variables`) -> `output_variables`.
    """

    models: List["ManyToOneImputation"]
    """Each column of the output variables is predicted by a separate model, stored in this list."""
    X_columns: List[str]
    """The names of the input variables."""
    Y_columns: List[str]
    """The names of the output variables."""
    random_generator: np.random.Generator = None
    """The random generator used to sample from the distribution of the imputation."""

    X_category_mappings: List[Dict[str, int]] = None
    """The mapping from category names to integers for each input variable."""

    def encode_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_category_mappings is None:
            self.X_category_mappings = {
                i: get_category_mapping(X[column])
                if X[column].dtype == "object"
                else None
                for i, column in enumerate(X.columns)
            }
        X = X.copy()
        for i, column in enumerate(X.columns):
            if self.X_category_mappings.get(i) is not None:
                X[column] = X[column].map(self.X_category_mappings[i])
        return X

    def train(self, X: pd.DataFrame, Y: pd.DataFrame, num_trees: int = 100):
        """
        Train a random forest model to predict the output variables from the input variables.

        Args:
            X (pd.DataFrame): The dataset containing the input variables.
            Y (pd.DataFrame): The dataset containing the output variables.
        """

        self.X_columns = X.columns
        self.Y_columns = Y.columns

        X = self.encode_categories(X)

        self.models = []
        # We train a separate model for each output variable. For example, if X = [income, age] and Y = [height, weight], we train two models:
        # 1. Predict height from income and age.
        # 2. Predict weight from income, age and (predicted) height.

        for i in tqdm(range(len(Y.columns)), desc="Training models"):
            Y_columns = Y.columns[:i]
            if i == 0:
                X_ = to_array(X)
            else:
                X_ = to_array(pd.concat([X, Y[Y_columns]], axis=1))
            y_ = to_array(Y[Y.columns[i]])
            model = ManyToOneImputation()
            model.encode_categories = self.encode_categories
            model.train(X_, y_, num_trees=num_trees)
            self.models.append(model)

    def predict(
        self,
        X: pd.DataFrame,
        mean_quantile: float = 0.5,
        verbose: bool = False,
        fixed_quantile: float = None,
    ) -> pd.DataFrame:
        """
        Predict the output variables for the input dataset.

        Args:
            X (pd.DataFrame): The dataset to predict on.
            mean_quantile (float): The beta parameter for the imputation.

        Returns:
            pd.DataFrame: The predicted dataset.
        """

        if isinstance(X, list):
            X = pd.DataFrame(X, columns=self.X_columns)

        X = pd.DataFrame(X, columns=self.X_columns)

        if self.random_generator is None:
            self.random_generator = np.random.default_rng()
        X = to_array(self.encode_categories(X))
        Y = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            if verbose:
                print(f"Imputing {self.Y_columns[i]}...")
            if mean_quantile is None and fixed_quantile is not None:
                m_quantile = None
                f_quantile = fixed_quantile
            if isinstance(mean_quantile, list):
                m_quantile = mean_quantile[i]
                f_quantile = None
            else:
                m_quantile = mean_quantile
                f_quantile = None
            X_ = np.concatenate([X, Y[:, :i]], axis=1)
            model.encode_categories = self.encode_categories
            Y[:, i] = model.predict(X_, mean_quantile=m_quantile, random_generator=self.random_generator, fixed_quantile=f_quantile)
        return pd.DataFrame(Y, columns=self.Y_columns)

    def save(self, path: str):
        """
        Save the imputation model to disk.

        Args:
            path (str): The path to save the model to.
        """

        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            # Store the models only in a dictionary.
            data = dict(
                models=self.models,
                X_columns=self.X_columns,
                X_category_mappings=self.X_category_mappings,
                Y_columns=self.Y_columns,
            )
            pickle.dump(data, f)

    @staticmethod
    def load(path: str) -> "Imputation":
        """
        Load the imputation model from disk.

        Args:
            path (str): The path to load the model from.

        Returns:
            Imputation: The imputation model.
        """

        import pickle

        imputation = Imputation()
        with open(path, "rb") as f:
            data = pickle.load(f)
            imputation.models = data["models"]
            imputation.X_columns = data["X_columns"]
            imputation.X_category_mappings = data["X_category_mappings"]
            imputation.Y_columns = data["Y_columns"]
            for model in imputation.models:
                model.encode_categories = imputation.encode_categories
                model.X_category_mappings = imputation.X_category_mappings
        return imputation

    def solve_for_mean_quantiles(
        self, targets: list, input_data: pd.DataFrame, weights: pd.Series
    ):
        mean_quantiles = []
        input_data = input_data.copy()
        for i, model in enumerate(self.models):
            mean_quantiles.append(
                model.solve_for_mean_quantile(
                    target=targets[i],
                    input_df=input_data,
                    weights=weights,
                    verbose=True,
                )
            )
            predicted_column = model.predict(input_data, mean_quantiles[-1])
            input_data[self.Y_columns[i]] = predicted_column
        return mean_quantiles


class ManyToOneImputation:
    """
    An `Imputation` consists of a set of `ManyToOneImputation` models, one for each output variable.
    """

    model: RandomForestRegressor
    """The random forest model."""

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series = None,
        num_trees: int = 100,
    ):
        """
        Train a random forest model to predict the output variable from the input variables.

        Args:
            X (pd.DataFrame): The dataset containing the input variables.
            y (pd.Series): The dataset containing the output variable.
            sample_weight (pd.Series): The sample weights.
        """

        X = to_array(X)
        y = to_array(y)
        self.model = RandomForestRegressor(
            n_estimators=num_trees, bootstrap=True, max_samples=0.01
        )
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(
        self,
        X: pd.DataFrame,
        mean_quantile: float = 0.5,
        random_generator: np.random.Generator = None,
        fixed_quantile: float = None,
    ) -> pd.DataFrame:
        """
        Predict the output variable for the input dataset.

        Args:
            X (pd.DataFrame): The dataset to predict on.
            mean_quantile (float): The mean quantile under the Beta distribution.
            random_generator (np.random.Generator): The random generator.

        Returns:
            pd.Series: The predicted distribution of values for each input row.
        """
        if isinstance(X, pd.DataFrame) and any(
            [X[column].dtype == "O" for column in X.columns]
        ):
            X = self.encode_categories(X)
        X = to_array(X)
        tree_predictions = [tree.predict(X) for tree in self.model.estimators_]

        # Get the percentiles of the predictions.
        tree_predictions = np.array(tree_predictions).transpose()
        if mean_quantile is None and fixed_quantile is not None:
            input_quantiles = np.full(
                tree_predictions.shape[0], fixed_quantile
            )
        else:
            if mean_quantile is None:
                mean_quantile = 0.5
            a = mean_quantile / (1 - mean_quantile)
            if random_generator is None:
                random_generator = np.random.default_rng()
            input_quantiles = random_generator.beta(
                a, 1, size=tree_predictions.shape[0]
            )
        x = np.apply_along_axis(
            lambda x: np.percentile(x[1:], x[0]),
            1,
            np.concatenate(
                [
                    np.array(input_quantiles)[:, np.newaxis] * 100,
                    tree_predictions,
                ],
                axis=1,
            ),
        )
        return x

    def solve_for_mean_quantile(
        self,
        target: float,
        input_df: pd.DataFrame,
        weights: np.ndarray,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Solve for the mean quantile that produces the target value.

        Args:
            target (float): The target value.
            input_df (pd.DataFrame): The input dataset.
            weights (np.ndarray): The sample weights.
            max_iterations (int, optional): The maximum number of iterations. Defaults to 5.
            verbose (bool, optional): Whether to print the loss at each iteration. Defaults to False.

        Returns:
            float: The mean quantile.
        """

        def loss(mean_quantile):
            pred_values = self.predict(input_df, mean_quantile)
            pred_aggregate = (pred_values * weights).sum()
            print(
                f"PREDICTED: {pred_aggregate/1e9:.1f} (target: {target/1e9:.1f})"
            )
            return (pred_aggregate - target) ** 2, pred_aggregate

        best_loss = float("inf")
        min_quantile = 0
        max_quantile = 1

        # Binary search for the mean quantile.
        for i in range(max_iterations):
            mean_quantile = (min_quantile + max_quantile) / 2
            loss_value, pred_agg = loss(mean_quantile)
            if verbose:
                print(
                    f"Iteration {i}: {mean_quantile:.4f} (loss: {loss_value:.4f})"
                )
            if loss_value < best_loss:
                best_loss = loss_value
            if pred_agg < target:
                min_quantile = mean_quantile
            else:
                max_quantile = mean_quantile
        return mean_quantile
