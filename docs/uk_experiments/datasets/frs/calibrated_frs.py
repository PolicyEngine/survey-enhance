from survey_enhance.dataset import Dataset
from survey_enhance.reweight import CalibratedWeights
from loss.loss import Loss, calibration_parameters
from loss.categories.country_level_programs import IncomeTax
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
    output_year: int
    epochs: int = 100
    learning_rate: float = 1e0
    log_dir: str = None

    @staticmethod
    def from_dataset(
        dataset: Type[Dataset],
        year: int = 2022,
        out_year: int = 2022,
        force_generate: bool = False,
        force_not_generate: bool = False,
        log_folder: str = None,
        verbose: bool = False,
    ):
        class CalibratedFRSFromDataset(CalibratedFRS):
            name = f"calibrated_{dataset.name}"
            label = f"Calibrated {dataset.label} {year} {out_year}"
            input_dataset = dataset
            input_dataset_year = year
            output_year = out_year
            log_dir = log_folder
            file_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / f"calibrated_{dataset.name}.h5"
            )
            data_format = dataset.data_format

        dataset = CalibratedFRSFromDataset()
        if not force_not_generate and (force_generate or not dataset.exists):
            dataset.generate(verbose=verbose)

        return CalibratedFRSFromDataset

    def generate(self, verbose: bool = False):
        from .frs import OutputDataset

        input_dataset = OutputDataset.from_dataset(self.input_dataset)()

        original_weights = input_dataset.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            input_dataset,
            IncomeTax,
            calibration_parameters,
        )
        print("Calibrating weights...")
        weights = calibrated_weights.calibrate(
            "2022-01-01",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            verbose=verbose,
            log_dir=self.log_dir,
        )

        data = self.input_dataset().load_dataset()

        data["household_weight"] = weights

        self.save_dataset(data)
