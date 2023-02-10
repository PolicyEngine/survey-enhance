import torch
import pandas as pd
from policyengine_core.parameters import ParameterNodeAtInstant, ParameterNode
from typing import List, Type, Tuple, Dict
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from survey_enhance.dataset import Dataset
import warnings


class LossCategory(torch.nn.Module):
    """
    A loss category is essentially a loss function, but contains a number of utilities for ease of programming, like
    decomposition into weighted and normalised subcategories, and logging.
    """

    weight: float = 1.0
    """The weight of this loss category in the total loss."""
    subcategories: List[Type["LossCategory"]] = []
    """The subcategories of this loss category."""
    static_dataset = False
    """Whether the dataset is static, i.e. does not change between epochs."""

    normalise: bool = True
    """Whether to normalise the starting loss value to 1."""

    diagnostic: bool = False
    """Whether to log the full tree of losses."""

    diagnostic_tree: Dict[str, float] = None
    """The tree of losses."""

    def __init__(
        self,
        dataset: Dataset,
        calibration_parameters: ParameterNodeAtInstant,
        weight: float = None,
        ancestor: "LossCategory" = None,
        static_dataset: bool = None,
        comparison_white_list: List[str] = None,
        comparison_black_list: List[str] = None,
        name: str = None,
        normalise: bool = None,
        diagnostic: bool = None,
    ):
        super().__init__()
        if weight is not None:
            self.weight = weight

        if normalise is not None:
            self.normalise = normalise

        if diagnostic is not None:
            self.diagnostic = diagnostic

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

        self.name = (
            name + "." + self.__class__.__name__
            if name is not None
            else self.__class__.__name__
        )

        self.sublosses = torch.nn.ModuleList(
            [
                subcategory(
                    dataset,
                    calibration_parameters,
                    ancestor=self.ancestor,
                    static_dataset=self.static_dataset,
                    comparison_white_list=self.comparison_white_list,
                    comparison_black_list=self.comparison_black_list,
                    name=self.name,
                    diagnostic=self.diagnostic,
                )
                for subcategory in self.subcategories
            ]
        )

        def filtered_get_comparisons(dataset: Dataset):
            comparisons = self.get_comparisons(dataset)
            if self.comparison_white_list is not None:
                comparisons = [
                    comparison
                    for comparison in comparisons
                    if comparison[0] in self.comparison_white_list
                ]
            if self.comparison_black_list is not None:
                comparisons = [
                    comparison
                    for comparison in comparisons
                    if comparison[0] not in self.comparison_black_list
                ]
            return comparisons

        self._get_comparisons = filtered_get_comparisons

    def create_holdout_sets(
        self,
        dataset: Dataset,
        num_sets: int,
        num_weights: int,
        exclude_by_name: str = None,
    ) -> List[Tuple[Dataset, Dataset]]:
        # Run the loss function, get the list of all comparisons, then split into holdout sets

        comparisons = self.collect_comparison_log()
        if len(comparisons) == 0:
            household_weight = torch.tensor(
                0 * np.zeros(num_weights),
                requires_grad=True,
            )
            self.forward(household_weight, dataset, initial_run=True)
            comparisons = self.collect_comparison_log()

        comparisons_name_filter = (
            ~comparisons.full_name.str.contains(exclude_by_name)
            if exclude_by_name is not None
            else pd.Series([True] * len(comparisons))
        )

        individual_comparisons = pd.Series(
            comparisons[
                (comparisons.type == "individual") & comparisons_name_filter
            ].name.unique()
        )
        individual_comparisons = individual_comparisons.sample(
            frac=1
        ).reset_index(drop=True)
        individual_comparisons = individual_comparisons.groupby(
            np.arange(len(individual_comparisons)) % num_sets
        ).apply(lambda x: x.tolist())
        return individual_comparisons.tolist()

    def get_comparisons(
        self, dataset: Dataset
    ) -> List[Tuple[str, float, torch.Tensor]]:
        raise NotImplementedError(
            f"Loss category {self.__class__.__name__} does not implement an evaluation method."
        )

    def collect_comparison_log(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.comparison_log,
            columns=[
                "epoch",
                "name",
                "y_true",
                "y_pred",
                "loss",
                "type",
                "full_name",
            ],
        )
        for subloss in self.sublosses:
            df = df.append(subloss.collect_comparison_log())
        return df

    def evaluate(
        self, household_weights: torch.Tensor, dataset: Dataset
    ) -> torch.Tensor:
        if self.static_dataset and self.comparisons is not None:
            comparisons = self.comparisons
        else:
            comparisons = self._get_comparisons(dataset)
            if self.static_dataset:
                self.comparisons = comparisons

        loss = torch.tensor(1e-3)
        for comparison in comparisons:
            if len(comparison) == 3:
                name, y_pred_array, y_true = comparison
                weight = 1
            elif len(comparison) == 4:
                name, y_pred_array, y_true, weight = comparison
            # y_pred_array needs to be a weighted sum with household_weights
            y_pred_array = torch.tensor(
                np.array(y_pred_array).astype(float), requires_grad=True
            )
            y_pred = torch.sum(y_pred_array * household_weights)
            BUFFER = 1e4
            loss_addition = (
                (y_pred + BUFFER) / (y_true + BUFFER) - 1
            ) ** 2 * weight
            if torch.isnan(loss_addition):
                raise ValueError(
                    f"Loss for {name} is NaN (y_pred={y_pred}, y_true={y_true})"
                )
            loss = loss + loss_addition
            self.comparison_log.append(
                (
                    self.ancestor.epoch,
                    name,
                    y_true,
                    float(y_pred),
                    float(loss_addition),
                    "individual",
                    self.name,
                )
            )
        return loss

    def forward(
        self,
        household_weights: torch.Tensor,
        dataset: Dataset,
        initial_run: bool = False,
    ) -> torch.Tensor:
        if torch.isnan(household_weights).any():
            raise ValueError("NaN in household weights")
        if self.initial_loss_value is None and not initial_run:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.initial_loss_value = torch.tensor(
                    self.forward(household_weights, dataset, initial_run=True),
                    requires_grad=False,
                )

        if not initial_run:
            self.epoch += 1

        loss = torch.tensor(
            0.0, requires_grad=True
        )  # To avoid division by zero

        try:
            self_loss = self.evaluate(household_weights, dataset)
            loss = loss + self_loss
        except NotImplementedError:
            pass

        if any(subloss.weight is None for subloss in self.sublosses):
            sublosses_str = "\n  - " + "\n  - ".join(
                [
                    subloss.__class__.__name__
                    for subloss in self.sublosses
                    if subloss.weight is None
                ]
            )
            raise ValueError(
                f"Loss category {self.__class__.__name__} has sublosses with no weight. These are: {sublosses_str}"
            )
        total_subloss_weight = sum(
            subloss.weight for subloss in self.sublosses
        )
        for subloss in self.sublosses:
            subcategory_loss = (
                subloss(household_weights, dataset)
                * subloss.weight
                / total_subloss_weight
            )
            self.comparison_log.append(
                (
                    self.ancestor.epoch,
                    subloss.__class__.__name__,
                    0,
                    0,
                    float(subcategory_loss) * subloss.weight,
                    "category",
                    subloss.name,
                )
            )
            if self.diagnostic:
                if self.diagnostic_tree is None:
                    self.diagnostic_tree = {}
                self.diagnostic_tree[subloss.name] = dict(
                    loss=float(subcategory_loss),
                    children=subloss.diagnostic_tree,
                )
            loss = loss + subcategory_loss

        if initial_run or not self.normalise:
            return loss
        else:
            return loss / self.initial_loss_value


class CalibratedWeights:
    dataset: Dataset
    initial_weights: np.ndarray
    calibration_parameters: ParameterNode
    loss_type: Type[torch.nn.Module]

    def __init__(
        self,
        initial_weights: np.ndarray,
        dataset: Dataset,
        loss_type: Type[torch.nn.Module],
        calibration_parameters: ParameterNode,
    ):
        self.initial_weights = initial_weights
        self.dataset = dataset
        self.loss_type = loss_type
        self.calibration_parameters = calibration_parameters

    def calibrate(
        self,
        time_instant: str,
        epochs: int = 1_000,
        learning_rate: float = 1e-1,
        validation_split: float = 0.0,
        validation_blacklist: List[str] = None,
        rotate_holdout_sets: bool = False,
        log_dir: str = None,
        tensorboard_log_dir: str = None,
        log_frequency: int = 100,
        verbose: bool = False,
    ) -> np.ndarray:
        self.verbose = verbose
        calibration_parameters_at_instant = self.calibration_parameters(
            time_instant
        )
        self.loss = loss = self.loss_type(
            self.dataset,
            calibration_parameters_at_instant,
            static_dataset=True,
        )

        if tensorboard_log_dir is not None:
            tensorboard_log_dir = Path(tensorboard_log_dir)
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            writer = None

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_df = pd.DataFrame()
        else:
            log_df = None

        if validation_split > 0:
            if validation_blacklist is None:
                validation_blacklist = []
            num_holdout_sets = int(1 / validation_split)
            holdout_sets = loss.create_holdout_sets(
                self.dataset,
                num_holdout_sets,
                len(self.initial_weights),
                exclude_by_name="Demographics",
            )
            train_loss_fn = self.loss_type(
                self.dataset,
                calibration_parameters_at_instant,
                static_dataset=True,
                comparison_black_list=holdout_sets[0],
            )
            validation_loss_fn = self.loss_type(
                self.dataset,
                calibration_parameters_at_instant,
                static_dataset=True,
                comparison_white_list=holdout_sets[0],
            )
        else:
            holdout_sets = None
            train_loss_fn = loss
            validation_loss_fn = None

        if rotate_holdout_sets:
            for i in range(len(holdout_sets)):
                weights = self._train(
                    train_loss_fn,
                    validation_loss_fn,
                    epochs,
                    learning_rate,
                    log_df,
                    log_dir,
                    writer,
                    log_frequency,
                    i,
                )
        else:
            weights = self._train(
                train_loss_fn,
                validation_loss_fn,
                epochs,
                learning_rate,
                log_df,
                log_dir,
                writer,
                log_frequency,
            )

        return weights

    def _train(
        self,
        training_loss_fn: LossCategory,
        validation_loss_fn: LossCategory,
        epochs: int,
        learning_rate: float,
        log_df: pd.DataFrame = None,
        log_dir: Path = None,
        tensorboard_log_writer: SummaryWriter = None,
        log_every: int = 100,
        holdout_set_index: int = None,
    ) -> np.ndarray:
        household_weights = torch.tensor(
            self.initial_weights, requires_grad=True
        )
        weight_adjustment = torch.tensor(
            np.zeros_like(self.initial_weights), requires_grad=True
        )
        optimizer = torch.optim.Adam([weight_adjustment], lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = training_loss_fn(
                household_weights + weight_adjustment, self.dataset
            )
            loss.backward()
            optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch}: {loss.item()}")

            if log_df is not None and epoch % log_every == 0:
                training_log = training_loss_fn.collect_comparison_log()
                training_log["validation"] = False
                if holdout_set_index is not None:
                    training_log["holdout_set"] = holdout_set_index
                if validation_loss_fn is not None:
                    validation_log = (
                        validation_loss_fn.collect_comparison_log()
                    )
                    validation_log["validation"] = True
                    if holdout_set_index is not None:
                        validation_log["holdout_set"] = holdout_set_index
                else:
                    validation_log = pd.DataFrame()
                log_df = pd.concat([log_df, training_log, validation_log])
                log_df.to_csv(log_dir / "calibration_log.csv", index=False)

                if tensorboard_log_writer is not None:
                    epoch_df = log_df[log_df["epoch"] == epoch]
                    for loss_name in epoch_df["name"].unique():
                        loss_df = epoch_df[epoch_df["name"] == loss_name]
                        if len(loss_df) > 0:
                            validation_status = (
                                "training"
                                if not loss_df["validation"].unique()[0]
                                else "validation"
                            )
                            metric_type = loss_df["type"].unique()[0]
                            tensorboard_log_writer.add_scalar(
                                f"loss/{loss_name}/{validation_status}",
                                loss_df["loss"].mean(),
                                epoch,
                            )
                            if metric_type == "individual":
                                # Then we also have y_pred and y_true
                                y_pred_value = loss_df["y_pred"].mean()
                                y_true_value = loss_df["y_true"].mean()
                                tensorboard_log_writer.add_scalar(
                                    f"model/{loss_name}/{validation_status}",
                                    y_pred_value,
                                    epoch,
                                )
                                tensorboard_log_writer.add_scalar(
                                    f"target/{loss_name}/{validation_status}",
                                    y_true_value,
                                    epoch,
                                )

        return (household_weights + weight_adjustment).detach().numpy()
