from loss.loss import Loss, calibration_parameters
from datasets.frs import (
    FRS_2019_20,
    SPIEnhancedFRS2019_20,
    CalibratedFRS,
    PercentileMatchedFRS,
)
from datasets.output_dataset import OutputDataset
import torch

datasets = {}

datasets["Original FRS"] = OutputDataset.from_dataset(
    FRS_2019_20, 2019, 2022
)()
datasets["Calibrated FRS"] = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(FRS_2019_20, 2019, 2022, force_generate=True, verbose=True),
    force_generate=True,
)()
print("Calibrated FRS done")
datasets["Calibrated SPI-enhanced FRS"] = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(
        SPIEnhancedFRS2019_20, 2019, 2022, force_generate=True, verbose=True
    ),
    force_generate=True,
)()

loss = Loss(
    datasets["Original FRS"],
    calibration_parameters(f"2022-01-01"),
    static_dataset=False,
)

losses = {}
for name, dataset in datasets.items():
    print(f"Calculating loss for {name}...", flush=True)
    losses[name] = loss(
        torch.tensor(dataset.household.household_weight.values), dataset
    )

for name, loss in losses.items():
    print(f"{name}: {loss}")
