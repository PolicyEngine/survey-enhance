import torch
import pandas as pd
from policyengine_core.parameters import ParameterNodeAtInstant
from typing import List, Type, Tuple
import numpy as np
from survey_enhance.dataset import Dataset


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
        df = pd.DataFrame(self.comparison_log, columns=["epoch", "name", "y_true", "y_pred", "loss"])
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
            loss_addition = torch.abs(y_true - y_pred) ** 2
            loss += loss_addition
            self.comparison_log.append((self.epoch, name, y_true, float(y_pred), float(loss_addition)))
        self.epoch += 1
        return loss
    
    def forward(self, household_weights: torch.Tensor, dataset: Dataset, initial_run: bool = False) -> torch.Tensor:
        if self.initial_loss_value is None and not initial_run:
            self.initial_loss_value = self.forward(household_weights, dataset, initial_run=True)
        
        loss = torch.tensor(1.) # To avoid division by zero

        try:
            self_loss = self.evaluate(household_weights, dataset)
            loss += self_loss
        except NotImplementedError:
            pass
        
        if any(subloss.weight is None for subloss in self.sublosses):
            sublosses_str = "\n  - " + '\n  - '.join([subloss.__class__.__name__ for subloss in self.sublosses if subloss.weight is None])
            raise ValueError(f"Loss category {self.__class__.__name__} has sublosses with no weight. These are: {sublosses_str}")
        total_subloss_weight = sum(subloss.weight for subloss in self.sublosses)
        for subloss in self.sublosses:
            subcategory_loss = subloss(household_weights, dataset) / total_subloss_weight
            self.comparison_log.append(
                (self.epoch, subloss.__class__.__name__, 0, 0, float(subcategory_loss))
            )
            loss += subcategory_loss
        
        if initial_run:
            return loss
        else:
            return (loss / self.initial_loss_value) * self.weight