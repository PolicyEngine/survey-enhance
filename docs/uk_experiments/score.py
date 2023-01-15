from loss.loss import Loss, calibration_parameters
from datasets.frs import FRS_2020_OUT_22, FRS_2018_21_OUT_22, CalibratedFRS_2020_21_22, CalibratedFRS_2018_21_22
import torch

dataset = FRS_2020_OUT_22()
pooled_dataset = CalibratedFRS_2018_21_22()
loss = Loss(
    dataset,
    calibration_parameters(f"2022-01-01"),
    static_dataset=False,
    normalise=False,
    diagnostic=True,
)

original_weights = torch.tensor(dataset.household.household_weight.values)
pooled_weights = torch.tensor(pooled_dataset.household.household_weight.values)

original_loss = loss(original_weights, dataset)
pooled_loss = loss(pooled_weights, pooled_dataset)

print(f"Original loss: {original_loss}")
print(f"Pooled loss: {pooled_loss}")