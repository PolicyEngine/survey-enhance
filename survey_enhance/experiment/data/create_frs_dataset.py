from typing import Dict, Any
from survey_enhance.dataset import Dataset

def create_frs_dataset(
    overrides: Dict[str, Any],
) -> Dataset:
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
        "income_tax",
        "tax_band",
        "region",
        "country",
        "adjusted_net_income",
        "employment_status",
    ]

    BENUNIT_COLUMNS = [
        "benunit_id",
        "household_id",
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
        "household_weight",
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
        "income_tax",
    ]

    for variable_name, value in overrides.items():
        frs_simulation.set_input(variable_name, 2022, value)

    frs_person_df = frs_simulation.calculate_dataframe(PERSON_COLUMNS, period=2022)
    frs_benunit_df = frs_simulation.calculate_dataframe(BENUNIT_COLUMNS, period=2022)
    frs_household_df = frs_simulation.calculate_dataframe(HOUSEHOLD_COLUMNS, period=2022)

    for personal_variable in PERSON_COLUMNS:
        # Add a participants column to the household dataframe, if numeric
        if "float" in str(frs_person_df[personal_variable].dtype) or "int" in str(frs_person_df[personal_variable].dtype):
            frs_person_df[f"{personal_variable}_participants"] = frs_person_df[personal_variable] > 0
            frs_household_df[f"{personal_variable}_participants"] = frs_person_df.groupby("person_household_id")[f"{personal_variable}_participants"].transform("sum")

    for benunit_variable in BENUNIT_COLUMNS:
        # Add a participants column to the household dataframe, if numeric
        if "float" in str(frs_benunit_df[benunit_variable].dtype) or "int" in str(frs_benunit_df[benunit_variable].dtype):
            frs_benunit_df[f"{benunit_variable}_participants"] = frs_benunit_df[benunit_variable] > 0
            frs_household_df[f"{benunit_variable}_participants"] = frs_benunit_df.groupby("household_id")[f"{benunit_variable}_participants"].transform("sum")

    for household_variable in HOUSEHOLD_COLUMNS:
        # Add a participants column to the household dataframe, if numeric
        if ("float" in str(frs_household_df[household_variable].dtype) or "int" in str(frs_household_df[household_variable].dtype)) and "_participants" not in household_variable:
            frs_household_df[f"{household_variable}_participants"] = frs_household_df[household_variable] > 0

    return Dataset(
        person_df=frs_person_df,
        benunit_df=frs_benunit_df,
        household_df=frs_household_df,
    )