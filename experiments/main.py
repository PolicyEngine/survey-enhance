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
    "tenure_type",
    "council_tax_band",
]

frs_person_df = frs_simulation.calculate_dataframe(PERSON_COLUMNS, period=2022)
frs_benunit_df = frs_simulation.calculate_dataframe(BENUNIT_COLUMNS, period=2022)
frs_household_df = frs_simulation.calculate_dataframe(HOUSEHOLD_COLUMNS, period=2022)
