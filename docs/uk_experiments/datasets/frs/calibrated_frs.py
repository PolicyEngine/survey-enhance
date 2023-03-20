from policyengine_core.data import Dataset
from survey_enhance.reweight import CalibratedWeights
from loss.loss import Loss, calibration_parameters
import numpy as np
from pathlib import Path
from typing import Type
import pandas as pd


def sum_by_household(values: pd.Series, dataset: Dataset) -> np.ndarray:
    return (
        pd.Series(values)
        .groupby(dataset.person.person_household_id.values)
        .sum()
        .values
    )


class CalibratedFRS(Dataset):
    input_dataset: Type[Dataset]
    input_dataset_year: int
    epochs: int = 256
    learning_rate: float = 2e1
    log_dir: str = None
    time_period: str = None
    log_verbose: bool = False

    @staticmethod
    def from_dataset(
        dataset: Type[Dataset],
        new_name: str = None,
        new_label: str = None,
        year: int = None,
        out_year: int = 2022,
        log_folder: str = None,
        verbose: bool = True,
    ):
        class CalibratedFRSFromDataset(CalibratedFRS):
            name = new_name
            label = new_label
            input_dataset = dataset
            input_dataset_year = year or dataset.time_period
            time_period = out_year
            log_dir = log_folder
            file_path = (
                Path(__file__).parent.parent.parent / "data" / f"{new_name}.h5"
            )
            data_format = dataset.data_format
            log_verbose = verbose

        return CalibratedFRSFromDataset

    def generate(self):
        from .frs import OutputDataset

        input_dataset = OutputDataset.from_dataset(self.input_dataset)()

        original_weights = input_dataset.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            input_dataset,
            Loss,
            calibration_parameters,
        )
        weights = calibrated_weights.calibrate(
            "2022-01-01",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            verbose=self.log_verbose,
            log_dir=self.log_dir,
        )

        data = self.input_dataset().load_dataset()

        data["household_weight"] = weights

        self.save_dataset(data)
