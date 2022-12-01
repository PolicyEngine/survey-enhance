# This Python script reproduces the results in the paper.

try:
    from policyengine_uk import Microsimulation, FRS, RawFRS, SPI, RawSPI
except ImportError:
    raise ImportError(
        f"Could not import policyengine_uk. Please install it with `pip install policyengine_uk`."
    )

from pathlib import Path

if 2019 not in RawFRS.years:
    DATA_FOLDER = Path(__file__).parent / "data" / "frs_2019_tab"
    RawFRS.generate(2019, DATA_FOLDER)
if 2019 not in FRS.years:
    FRS.generate(2019)
if 2022 not in FRS.years:
    FRS.generate(2022)
if 2019 not in RawSPI.years:
    DATA_FOLDER = Path(__file__).parent / "data" / "spi_2019_tab"
    RawSPI.generate(2019, DATA_FOLDER)
if 2019 not in SPI.years:
    SPI.generate(2019)
if 2022 not in SPI.years:
    SPI.generate(2022)

frs_simulation = Microsimulation(dataset=FRS, dataset_year=2019)

PERSON_COLUMNS = [
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "age",
    "gender",
    "employment_income",
    "self_employment_income",
    "pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "state_pension",
    "total_NI",
    "tax_band",
    "region",
]

BENUNIT_COLUMNS = [
    "benunit_id",
    "child_benefit",
    "child_tax_credit",
    "working_tax_credit",
    "housing_benefit",
    "ESA_income",
    "housing_benefit",
    "income_support",
    "JSA_income",
    "pension_credit",
    "tax_credits",
    "universal_credit",
    "working_tax_credit",
]

HOUSEHOLD_COLUMNS = [
    "household_id",
    "region",
    "council_tax_less_benefit",
    "council_tax_band",
    "ons_tenure_type",
    "people",
    "child_benefit",
    "child_tax_credit",
    "working_tax_credit",
    "housing_benefit",
    "ESA_income",
    "housing_benefit",
    "income_support",
    "JSA_income",
    "pension_credit",
    "tax_credits",
    "universal_credit",
    "working_tax_credit",
    "employment_income",
    "self_employment_income",
    "pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "state_pension",
    "total_NI",
    "country",
]

frs_person_df = frs_simulation.calculate_dataframe(PERSON_COLUMNS, period=2022)
frs_benunit_df = frs_simulation.calculate_dataframe(BENUNIT_COLUMNS, period=2022)
frs_household_df = frs_simulation.calculate_dataframe(HOUSEHOLD_COLUMNS, period=2022)

for personal_variable in PERSON_COLUMNS:
    # Add a participants column to the household dataframe, if numeric
    if "float" in str(frs_person_df[personal_variable].dtype) or "int" in str(frs_person_df[personal_variable].dtype):
        frs_household_df[f"{personal_variable}_participants"] = (
            frs_person_df.groupby("person_household_id")[personal_variable]
            .sum()
            .astype("int")
        )

for benunit_variable in BENUNIT_COLUMNS:
    # Add a participants column to the household dataframe, if numeric
    if "float" in str(frs_benunit_df[benunit_variable].dtype) or "int" in str(frs_benunit_df[benunit_variable].dtype):
        frs_household_df[f"{benunit_variable}_participants"] = (
            frs_benunit_df.groupby("benunit_id")[benunit_variable]
            .sum()
            .astype("int")
        )

from survey_enhance.experiments.loss.loss import Loss
from survey_enhance.loss import Dataset

dataset = Dataset(
    person_df=frs_person_df,
    benunit_df=frs_benunit_df,
    household_df=frs_household_df,
)

from policyengine_core.parameters import ParameterNode, uprate_parameters

calibration_parameters = ParameterNode(
    directory_path=Path(__file__).parent / "loss" / "calibration_parameters",
    name="calibration",
)

calibration_parameters = uprate_parameters(calibration_parameters)
calibration_parameters = calibration_parameters("2022-01-01").calibration

import torch

household_weights = torch.Tensor(frs_simulation.calculate("household_weight", 2019).values)

loss = Loss(dataset, calibration_parameters)
print(loss(household_weights, dataset))
print(loss(household_weights * 0, dataset))

import pandas as pd

epochs = []
names = []
y_true = []
y_pred = []

for epoch, name, y_t, y_p in loss.comparison_log:
    epochs.append(epoch)
    names.append(name)
    y_true.append(y_t)
    y_pred.append(y_p)

comparison_df = pd.DataFrame(
    {
        "epoch": epochs,
        "name": names,
        "y_true": y_true,
        "y_pred": y_pred,
    }
)

comparison_df.to_csv("comparison.csv")