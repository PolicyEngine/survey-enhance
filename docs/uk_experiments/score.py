from loss.loss import Loss, calibration_parameters
from datasets.frs import FRS_2019_20, SPIEnhancedFRS2019_20, CalibratedFRS
from datasets.output_dataset import OutputDataset
import torch

dataset = OutputDataset.from_dataset(FRS_2019_20, 2019, 2022)()
calibrated_dataset = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(SPIEnhancedFRS2019_20, 2022, 2022), 2022, 2022
)()
loss = Loss(
    dataset,
    calibration_parameters(f"2022-01-01"),
    static_dataset=False,
    normalise=False,
    diagnostic=True,
)

original_weights = torch.tensor(dataset.household.household_weight.values)
calibrated_weights = torch.tensor(
    calibrated_dataset.household.household_weight.values
)

original_loss = loss(original_weights, dataset)
calibrated_loss = loss(calibrated_weights, calibrated_dataset)

print(f"Original loss: {original_loss}")
print(f"Calibrated loss: {calibrated_loss}")
