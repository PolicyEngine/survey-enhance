from loss.loss import Loss, calibration_parameters
from datasets.frs import (
    FRS_2022,
    SPIEnhancedFRS_2022,
    CalibratedFRS,
    PercentileMatchedFRS,
)
from datasets.output_dataset import OutputDataset
import torch
from pathlib import Path

datasets = {}

datasets["Original FRS"] = OutputDataset.from_dataset(
    FRS_2022, force_generate=True
)()
datasets["SPI percentile-matched FRS"] = OutputDataset.from_dataset(
    PercentileMatchedFRS.from_dataset(
        FRS_2022,
        percentile_matched_variables=["dividend_income"],
        force_generate=True,
    ),
    force_generate=True,
)()

datasets["Calibrated FRS"] = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(FRS_2022, force_generate=True),
    force_generate=True,
)()
datasets["Calibrated SPI-enhanced FRS"] = OutputDataset.from_dataset(
    CalibratedFRS.from_dataset(
        SPIEnhancedFRS_2022,
        force_generate=True,
        log_folder=Path(__file__).parent / "logs",
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
    losses[name] = loss(
        torch.tensor(
            dataset.household.household_weight.values,
            device=torch.device("mps"),
        ),
        dataset,
    )

for name, loss in losses.items():
    print(f"{name}: {loss}")
