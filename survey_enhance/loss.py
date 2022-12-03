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

    def __init__(self, 
        dataset: Dataset, 
        calibration_parameters: ParameterNodeAtInstant, 
        weight: float = None, 
        ancestor: "LossCategory" = None, 
        static_dataset: bool = None,
        comparison_white_list: List[str] = None,
        comparison_black_list: List[str] = None,
        name: str = None,
    ):
        super().__init__()
        if weight is not None:
            self.weight = weight
        
        self.dataset = dataset
        self.calibration_parameters = calibration_parameters
        self.comparison_log = []
        self.initial_loss_value = None

        self.comparison_white_list = comparison_white_list
        self.comparison_black_list = comparison_black_list

        self.comparisons = None

        if ancestor is None:
            self.ancestor = self
        else:
            self.ancestor = ancestor
        
        self.epoch = 0
        if static_dataset is not None:
            self.static_dataset = static_dataset
        
        self.name = name + "." + self.__class__.__name__ if name is not None else self.__class__.__name__

        self.sublosses = torch.nn.ModuleList([
            subcategory(
                dataset, 
                calibration_parameters, 
                ancestor=self.ancestor, 
                static_dataset=self.static_dataset,
                comparison_white_list=self.comparison_white_list,
                comparison_black_list=self.comparison_black_list,
                name=self.name,
            )
            for subcategory in self.subcategories
        ])

        def filtered_get_comparisons(dataset: Dataset):
            comparisons = self.get_comparisons(dataset)
            if self.comparison_white_list is not None:
                comparisons = [comparison for comparison in comparisons if comparison[0] in self.comparison_white_list]
            if self.comparison_black_list is not None:
                comparisons = [comparison for comparison in comparisons if comparison[0] not in self.comparison_black_list]
            return comparisons
        
        self._get_comparisons = filtered_get_comparisons

    def create_holdout_sets(self, dataset: Dataset, num_sets: int, exclude_by_name: str = None) -> List[Tuple[Dataset, Dataset]]:
        # Run the loss function, get the list of all comparisons, then split into holdout sets

        comparisons = self.collect_comparison_log()
        if len(comparisons) == 0:
            household_weight = torch.tensor(0 * dataset.household_df.household_weight.values, requires_grad=True)
            self.forward(household_weight, dataset, initial_run=True)
            comparisons = self.collect_comparison_log()
        
        comparisons_name_filter = ~comparisons.full_name.str.contains(exclude_by_name) if exclude_by_name is not None else pd.Series([True] * len(comparisons))
        
        individual_comparisons = pd.Series(comparisons[(comparisons.type == "individual") & comparisons_name_filter].name.unique())
        individual_comparisons = individual_comparisons.sample(frac=1).reset_index(drop=True)
        individual_comparisons = individual_comparisons.groupby(np.arange(len(individual_comparisons)) % num_sets).apply(lambda x: x.tolist())
        return individual_comparisons.tolist()
    
    def get_comparisons(self, dataset: Dataset) -> List[Tuple[str, float, torch.Tensor]]:
        raise NotImplementedError(f"Loss category {self.__class__.__name__} does not implement an evaluation method.")

    def collect_comparison_log(self) -> pd.DataFrame:
        df = pd.DataFrame(self.comparison_log, columns=["epoch", "name", "y_true", "y_pred", "loss", "type", "full_name"])
        for subloss in self.sublosses:
            df = df.append(subloss.collect_comparison_log())
        return df
    
    def evaluate(self, household_weights: torch.Tensor, dataset: Dataset) -> torch.Tensor:
        if self.static_dataset and self.comparisons is not None:
            comparisons = self.comparisons
        else:
            comparisons = self._get_comparisons(dataset)
            if self.static_dataset:
                self.comparisons = comparisons

        loss = torch.tensor(1e-3)
        for name, y_pred_array, y_true in comparisons:
            # y_pred_array needs to be a weighted sum with household_weights
            y_pred_array = torch.tensor(np.array(y_pred_array).astype(float), requires_grad=True)
            y_pred = torch.sum(y_pred_array * household_weights)
            loss_addition = (y_pred / (y_true + 1) - 1) ** 2
            if torch.isnan(loss_addition):
                raise ValueError(f"Loss for {name} is NaN (y_pred={y_pred}, y_true={y_true})")
            loss = loss + loss_addition
            self.comparison_log.append((self.ancestor.epoch, name, y_true, float(y_pred), float(loss_addition), "individual", self.name + "." + self.__class__.__name__))
        return loss
    
    def forward(self, 
        household_weights: torch.Tensor, 
        dataset: Dataset, 
        initial_run: bool = False,
    ) -> torch.Tensor:
        if torch.isnan(household_weights).any():
            raise ValueError("NaN in household weights")
        if self.initial_loss_value is None and not initial_run:
            self.initial_loss_value = torch.tensor(self.forward(household_weights, dataset, initial_run=True), requires_grad=False)

        if not initial_run:
            self.epoch += 1
        
        loss = torch.tensor(0., requires_grad=True) # To avoid division by zero

        try:
            self_loss = self.evaluate(household_weights, dataset)
            loss = loss + self_loss
        except NotImplementedError:
            pass
        
        if any(subloss.weight is None for subloss in self.sublosses):
            sublosses_str = "\n  - " + '\n  - '.join([subloss.__class__.__name__ for subloss in self.sublosses if subloss.weight is None])
            raise ValueError(f"Loss category {self.__class__.__name__} has sublosses with no weight. These are: {sublosses_str}")
        total_subloss_weight = sum(subloss.weight for subloss in self.sublosses)
        for subloss in self.sublosses:
            subcategory_loss = subloss(household_weights, dataset) / total_subloss_weight
            self.comparison_log.append(
                (self.ancestor.epoch, subloss.__class__.__name__, 0, 0, float(subcategory_loss), "category", self.name)
            )
            loss = loss + subcategory_loss
        
        if initial_run:
            return loss
        else:
            return (loss / self.initial_loss_value) * self.weight