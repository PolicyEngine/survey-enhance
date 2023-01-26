from survey_enhance.dataset import Dataset
from survey_enhance.reweight import CalibratedWeights
from loss.loss import Loss, calibration_parameters
import numpy as np
from pathlib import Path
from typing import Type

class CalibratedFRS(Dataset):
    input_dataset: Type[Dataset]
    input_dataset_year: int
    output_year: int
    epochs: int = 400
    learning_rate: float = 25
    log_dir: str = None

    @staticmethod
    def from_dataset(dataset: Type[Dataset], year: int, out_year: int):
        class CalibratedFRSFromDataset(CalibratedFRS):
            name = f"calibrated_{dataset.name}_{year}_{out_year}"
            label = f"Calibrated {dataset.label} {year} {out_year}"
            input_dataset = dataset
            input_dataset_year = year
            output_year = out_year or 2023
            file_path = Path(__file__).parent / f"calibrated_{dataset.name}.h5"
            data_format = dataset.data_format

        dataset = CalibratedFRSFromDataset()
        dataset.generate()

        return CalibratedFRSFromDataset

    def generate(self):
        from .frs import OutputDataset

        input_dataset = OutputDataset.from_dataset(self.input_dataset, self.input_dataset_year, self.output_year)()

        original_weights = input_dataset.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            input_dataset,
            Loss,
            calibration_parameters,
        )

        weights = calibrated_weights.calibrate(
            "2022-01-01",
            epochs=400,
            learning_rate=25,
            log_dir="."
        )

        calibrated_weights.loss.collect_comparison_log().to_csv("log.csv")

        data = self.input_dataset().load_dataset()

        data["household_weight"] = weights

        self.save_dataset(data)
