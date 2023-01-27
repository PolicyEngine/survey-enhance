from survey_enhance.impute import Imputation
from survey_enhance.dataset import Dataset
from pathlib import Path


class SPIEnhancedFRS2019_20(Dataset):
    name = "spi_enhanced_frs_2019_20"
    label = "SPI-Enhanced FRS 2019/20"
    file_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "spi_enhanced_frs_2019_20.h5"
    )
    data_format = Dataset.ARRAYS

    def generate(self):
        from datasets.frs import FRS_2019_20
        from policyengine_uk import Microsimulation

        frs = FRS_2019_20().load()

        new_values = {}

        for variable in frs.keys():
            if "_id" in variable:
                # e.g. [1, 2, 3] -> [10, 20, 30, 11, 21, 31]
                values = list(frs[variable][...] * 10)
                values = [*values, *[x + 1 for x in values]]
                new_values[variable] = values
            elif "_weight" in variable:
                new_values[variable] = list(frs[variable][...]) + list(
                    frs[variable][...] * 0
                )
            else:
                new_values[variable] = list(frs[variable][...]) * 2

        TARGETS = [
            1.016e12,  # From up-to-date published RTI data
            123.3e9,  # This and below from the 2019-20 SPI, uprated by 16% (2019 -> 2022)
            7.25e9,  # 16% is the relative change from 2019 SPI total pay to 2022 EOY RTI total pay
            78.0e9,
            133.0e9,
            10.3e9,
            30.9e9,
            4.0e9,
            31.9e9,
        ]

        TARGETS = [0.5 for _ in TARGETS]

        income = Imputation.load("income.pkl")

        simulation = Microsimulation(
            dataset=FRS_2019_20(),
            dataset_year=2019,
        )

        input_df = simulation.calculate_dataframe(
            ["age", "gender", "region"], 2022
        )

        SOLVE_QUANTILES = False
        if SOLVE_QUANTILES:
            mean_quantiles = income.solve_quantiles(
                TARGETS,
                input_df,
                sim.calculate("household_weight", map_to="person").values,
            )
        else:
            mean_quantiles = [
                0.38,
                0.24,
                0.39,
                0.28,
                0.45,
                0.43,
                0.29,
                0.52,
                0.5,
            ]

        full_imputations = income.predict(input_df, mean_quantiles)
        for variable in full_imputations.columns:
            # Assign over the second half of the dataset
            if variable in new_values.keys():
                new_values[variable][
                    len(new_values[variable]) // 2 :
                ] = full_imputations[variable].values

        self.save_dataset(new_values)


IMPUTATIONS = [
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "pension_income",
    "employment_expenses",
    "property_income",
    "gift_aid",
    "pension_contributions",
]
