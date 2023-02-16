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
datasets["SPI percentile-matched FRS"] = OutputDataset.from_dataset(
    PercentileMatchedFRS.from_dataset(
        FRS_2019_20,
        percentile_matched_variables=["dividend_income"],
    ),
    2019,
    2022,
)()
datasets["Calibrated FRS"] = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(FRS_2019_20, 2019, 2022, force_generate=True),
    2022,
    2022,
    force_generate=True,
)()
datasets["Calibrated SPI-enhanced FRS"] = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(
        SPIEnhancedFRS2019_20, 2019, 2022, force_generate=True
    ),
    2022,
    2022,
    force_generate=True,
)()

loss = Loss(
    datasets["Original FRS"],
    calibration_parameters(f"2022-01-01"),
    static_dataset=False,
)

losses = {}
for name, dataset in datasets.items():
    losses[name] = loss(
        torch.tensor(dataset.household.household_weight.values), dataset
    )

for name, loss in losses.items():
    print(f"{name}: {loss}")
