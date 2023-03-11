from datasets.frs import (
    RawFRS,
    FRS,
    SPIEnhancedFRS,
    CalibratedFRS,
    StackedFRS,
    PercentileMatchedFRS,
)
from datasets.output_dataset import OutputDataset
from loss.loss import Loss, calibration_parameters

raw_frs_2019 = RawFRS.from_folder(
    "/Users/nikhil/ukda/frs_2019_20", "raw_frs_2019", "FRS 2019-20"
)

frs_2019 = FRS.from_dataset(
    raw_frs_2019,
    "frs_2019",
    "FRS 2019-20",
)

percentile_matched_frs = PercentileMatchedFRS.from_dataset(
    frs_2019,
    percentile_matched_variables=["dividend_income"],
)

raw_frs_2018 = RawFRS.from_folder(
    "/Users/nikhil/ukda/frs_2018_19", "raw_frs_2018", "FRS 2018-19"
)

frs_2018 = FRS.from_dataset(
    raw_frs_2018,
    "frs_2018",
    "FRS 2018-19",
)

raw_frs_2020 = RawFRS.from_folder(
    "/Users/nikhil/ukda/frs_2020_21", "raw_frs_2020", "FRS 2020-21"
)

frs_2020 = FRS.from_dataset(
    raw_frs_2020,
    "frs_2020",
    "FRS 2020-21",
)

pooled_frs_2018_20 = StackedFRS.from_dataset(
    [frs_2018, frs_2019, frs_2020],
    [0.0, 1.0, 0.0],
    "pooled_frs_2018_20",
    "FRS 2018-20",
)

spi_enhanced_pooled_frs_2018_20 = SPIEnhancedFRS.from_dataset(
    pooled_frs_2018_20,
    "spi_enhanced_pooled_frs_2018_20",
    "SPI-enhanced FRS 2018-20",
)

spi_enhanced_frs_2019 = SPIEnhancedFRS.from_dataset(
    frs_2019,
    "spi_enhanced_frs_2019",
    "SPI-enhanced FRS 2019-20",
)

calibrated_frs_2019 = CalibratedFRS.from_dataset(
    frs_2019,
    "calibrated_frs_2019",
    "Calibrated FRS 2019-20",
)

calibrated_spi_enhanced_frs_2019 = CalibratedFRS.from_dataset(
    spi_enhanced_frs_2019,
    "calibrated_spi_enhanced_frs_2019",
    "Calibrated SPI-enhanced FRS 2019-20",
)

enhanced_frs = CalibratedFRS.from_dataset(
    spi_enhanced_pooled_frs_2018_20,
    "enhanced_frs",
    "Calibrated SPI-enhanced FRS 2018-20",
    log_folder=".",
)

loss = Loss(
    frs_2019,
    calibration_parameters(f"2022-01-01"),
    static_dataset=False,
)


datasets = [
    frs_2019,
    percentile_matched_frs,
    frs_2018,
    frs_2020,
    pooled_frs_2018_20,
    spi_enhanced_frs_2019,
    calibrated_frs_2019,
    calibrated_spi_enhanced_frs_2019,
    spi_enhanced_pooled_frs_2018_20,
    enhanced_frs,
]

for dataset in datasets:
    data = OutputDataset.from_dataset(dataset)()
    loss_value = loss(
        data.household.household_weight,
        data,
    ).item()
    print(f"{dataset.label}: {loss_value}")
