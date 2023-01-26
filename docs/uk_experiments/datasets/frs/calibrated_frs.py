from survey_enhance.dataset import Dataset
from survey_enhance.reweight import CalibratedWeights
from loss.loss import Loss, calibration_parameters
import numpy as np
from pathlib import Path


class CalibratedFRS_2019_20_22(Dataset):
    name = "calibrated_frs_2019_20_22"
    label = "Calibrated FRS 2019_20"

    data_format = Dataset.ARRAYS
    file_path = Path(__file__).parent / "calibrated_frs_2019_20_22.h5"

    def generate(self):
        from .frs import FRS_2019_20, OutputDataset

        frs_2020_out_22 = OutputDataset.from_dataset(FRS_2019_20, 2020, 2022)()

        original_weights = frs_2020_out_22.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            frs_2020_out_22,
            Loss,
            calibration_parameters,
        )

        weights = calibrated_weights.calibrate(
            "2022-01-01",
            epochs=1_000,
            learning_rate=0.1,
        )

        data = FRS_2019_20().load_dataset()

        data["household_weight"] = weights

        self.save_dataset(data)
