from loss.loss import Loss, calibration_parameters
from datasets.frs import OutputFRS_2019_20_22, CalibratedFRS_2019_20_22
from datasets.output_dataset import OutputDataset
import torch

dataset = OutputFRS_2019_20_22()
calibrated_dataset = OutputDataset.from_dataset(
    CalibratedFRS_2019_20_22, 2022, 2022
)()
loss = Loss(
    dataset,
    calibration_parameters(f"2022-01-01"),
    static_dataset=False,
    normalise=False,
    diagnostic=True,
)

original_weights = torch.tensor(dataset.household.household_weight.values)
pooled_weights = torch.tensor(
    calibrated_dataset.household.household_weight.values
)

print(f"Original weights: {original_weights}")
print(f"Pooled weights: {pooled_weights}")

original_loss = loss(original_weights, dataset)
pooled_loss = loss(pooled_weights, calibrated_dataset)

print(f"Original loss: {original_loss}")
print(f"Pooled loss: {pooled_loss}")
