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
    epochs: int = 1_000
    learning_rate: float = 1e-1
    log_dir: str = None

    @staticmethod
    def from_dataset(
        dataset: Type[Dataset],
        year: int = 2022,
        out_year: int = 2022,
        force_generate: bool = False,
        force_not_generate: bool = False,
    ):
        class CalibratedFRSFromDataset(CalibratedFRS):
            name = f"calibrated_{dataset.name}_{year}_{out_year}"
            label = f"Calibrated {dataset.label} {year} {out_year}"
            input_dataset = dataset
            input_dataset_year = year
            output_year = out_year
            file_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / f"calibrated_{dataset.name}.h5"
            )
            data_format = dataset.data_format

        dataset = CalibratedFRSFromDataset()
        if not force_not_generate and (force_generate or not dataset.exists):
            dataset.generate()

        return CalibratedFRSFromDataset

    def generate(self):
        from .frs import OutputDataset

        input_dataset = OutputDataset.from_dataset(
            self.input_dataset, self.input_dataset_year, self.output_year
        )()

        original_weights = input_dataset.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            input_dataset,
            Loss,
            calibration_parameters,
        )

        weights = calibrated_weights.calibrate(
            "2022-01-01", epochs=256, learning_rate=1e1, log_dir="."
        )

        data = self.input_dataset().load_dataset()

        data["household_weight"] = weights

        self.save_dataset(data)
