import torch
import pandas as pd
from policyengine_core.parameters import ParameterNodeAtInstant
from typing import List, Type, Tuple
import numpy as np

class Dataset:
    person_df: pd.DataFrame
    benunit_df: pd.DataFrame
    household_df: pd.DataFrame

    def __init__(self, person_df: pd.DataFrame, benunit_df: pd.DataFrame, household_df: pd.DataFrame):
        self.person_df = person_df
        self.benunit_df = benunit_df
        self.household_df = household_df

class LossCategory(torch.nn.Module):
    weight: float = 1.
    subcategories: List[Type["LossCategory"]] = []
    static_dataset = False

    def __init__(self, dataset: Dataset, calibration_parameters: ParameterNodeAtInstant, weight: float = None):
        super().__init__()
        if weight is not None:
            self.weight = weight
        
        self.dataset = dataset
        self.calibration_parameters = calibration_parameters
        self.epoch = 0
        self.comparison_log = []
        self.initial_loss_value = None

        self.comparisons = None

        self.sublosses = torch.nn.ModuleList([
            subcategory(dataset, calibration_parameters)
            for subcategory in self.subcategories
        ])
    
    def get_comparisons(self, dataset: Dataset) -> List[Tuple[str, float, torch.Tensor]]:
        raise NotImplementedError(f"Loss category {self.__class__.__name__} does not implement an evaluation method.")

    def collect_comparison_log(self) -> pd.DataFrame:
        df = pd.DataFrame(self.comparison_log, columns=["epoch", "name", "y_true", "y_pred"])
        for subloss in self.sublosses:
            df = df.append(subloss.collect_comparison_log())
        return df
    
    def evaluate(self, household_weights: torch.Tensor, dataset: Dataset) -> torch.Tensor:
        if self.static_dataset and self.comparisons is not None:
            comparisons = self.comparisons
        else:
            comparisons = self.get_comparisons(dataset)
            if self.static_dataset:
                self.comparisons = comparisons

        loss = torch.tensor(0.)
        for name, y_pred_array, y_true in comparisons:
            # y_pred_array needs to be a weighted sum with household_weights
            y_pred_array = torch.Tensor(y_pred_array.astype(float))
            y_pred = torch.sum(y_pred_array * household_weights)
            loss += torch.abs(y_true - y_pred) ** 2
            self.comparison_log.append((self.epoch, name, y_true, float(y_pred)))
        self.epoch += 1
        return loss
    
    def forward(self, household_weights: torch.Tensor, dataset: Dataset, initial_run: bool = False) -> torch.Tensor:
        if self.initial_loss_value is None and not initial_run:
            self.initial_loss_value = self.forward(household_weights, dataset, initial_run=True)
        
        loss = torch.tensor(0.)

        try:
            subcategory_loss = self.evaluate(household_weights, dataset)
            loss += subcategory_loss
        except NotImplementedError:
            pass

        for subloss in self.sublosses:
            loss += subloss(household_weights, dataset) * subloss.weight
        
        if initial_run:
            return loss
        else:
            return (loss - self.initial_loss_value) * self.weight

def sum_by_household(values: pd.Series, dataset: Dataset) -> np.ndarray:
    return pd.Series(values).groupby(dataset.person_df.person_household_id).sum().values