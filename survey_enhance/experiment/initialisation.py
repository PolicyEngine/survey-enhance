from survey_enhance.experiment.loss.loss import Loss
from pathlib import Path
from survey_enhance.experiment.data.create_frs_dataset import create_frs_dataset

dataset = create_frs_dataset({})

from policyengine_core.parameters import ParameterNode, uprate_parameters

calibration_parameters = ParameterNode(
    directory_path=Path(__file__).parent / "loss" / "calibration_parameters",
    name="calibration",
)

calibration_parameters = uprate_parameters(calibration_parameters)
calibration_parameters = calibration_parameters("2022-01-01").calibration

import torch

household_weights = torch.tensor(dataset.household_df.household_weight.values, requires_grad=True)

loss = Loss(dataset, calibration_parameters)

