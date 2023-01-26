from survey_enhance.dataset import Dataset
from typing import Type
import pandas as pd
from pathlib import Path


class OutputDataset(Dataset):
    data_format = Dataset.TABLES
    input_dataset: Type[Dataset]
    input_dataset_year: int
    output_year: int

    @staticmethod
    def from_dataset(dataset: Type[Dataset], year: int, out_year: int):
        class OutputDatasetFromDataset(OutputDataset):
            name = f"{dataset.name}_{year}_{out_year}"
            label = f"{dataset.label} {year} {out_year}"
            input_dataset = dataset
            input_dataset_year = year
            output_year = out_year or 2023
            file_path = Path(__file__).parent / f"output_{dataset.name}.h5"

        dataset = OutputDatasetFromDataset()
        dataset.generate()

        return OutputDatasetFromDataset

    def generate(self):
        from policyengine_uk import Microsimulation

        if not self.input_dataset().exists:
            self.input_dataset().generate()

        sim = Microsimulation(
            dataset=self.input_dataset(), dataset_year=self.input_dataset_year
        )
        sim.default_calculation_period = self.output_year

        PERSON_VARIABLES = [
            "age",
            "gender",
            "region",
            "country",
            "person_id",
            "person_household_id",
            "tax_band",
            "adjusted_net_income",
        ]

        HOUSEHOLD_VARIABLES = [
            "household_id",
            "region",
            "country",
            "ons_tenure_type",
            "council_tax_band",
            "household_weight",
        ]

        PROGRAM_VARIABLES = [
            "income_support",
            "pension_credit",
            "working_tax_credit",
            "child_benefit",
            "child_tax_credit",
            "universal_credit",
            "state_pension",
            "total_NI",
            "JSA_income",
            "housing_benefit",
            "ESA_income",
            "employment_income",
            "self_employment_income",
            "pension_income",
            "property_income",
            "savings_interest_income",
            "council_tax_less_benefit",
            "dividend_income",
            "income_tax",
        ]

        variables = sim.tax_benefit_system.variables

        person = pd.DataFrame()

        for variable in PERSON_VARIABLES:
            person[variable] = sim.calculate(variable, map_to="person").values

        household = pd.DataFrame()

        for variable in HOUSEHOLD_VARIABLES:
            household[variable] = sim.calculate(variable).values

        for variable in PROGRAM_VARIABLES:
            if variables[variable].entity.key != "household":
                person[variable] = sim.calculate(
                    variable, map_to="person"
                ).values
                household[variable] = sim.calculate(
                    variable, map_to="household"
                ).values
                household[f"{variable}_participants"] = sim.map_result(
                    sim.calculate(variable).values > 0,
                    variables[variable].entity.key,
                    "household",
                )
            else:
                household[variable] = sim.calculate(variable).values
        self.save_dataset(
            dict(
                person=person,
                household=household,
            )
        )
