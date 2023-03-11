from survey_enhance.dataset import Dataset
from pathlib import Path
import pandas as pd
from datasets.frs.imputations.income import SPI_TAB_FOLDER, generate_spi_table
from survey_enhance.percentile_match import match_percentiles
from typing import List, Type
from microdf import MicroSeries


class PercentileMatchedFRS(Dataset):
    name = "percentile_matched_frs_2019_20"
    label = "Percentile Matched FRS"
    file_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / f"calibrated_percentile_matched_frs_2019_20.h5"
    )
    match_variables = [
        "employment_income",
        "self_employment_income",
        "pension_income",
        "dividend_income",
        "savings_interest_income",
    ]
    match_threshold = 0.95
    num_groups = 10

    @staticmethod
    def from_dataset(
        dataset: Type[Dataset],
        percentile_matched_variables: List[str] = None,
        threshold: float = 0.95,
        group_count: int = 10,
    ):
        class PercentileMatchedFRSFromDataset(PercentileMatchedFRS):
            name = f"percentile_matched_{dataset.name}"
            label = f"Percentile Matched {dataset.label}"
            input_dataset = dataset
            match_variables = (
                percentile_matched_variables
                if percentile_matched_variables is not None
                else PercentileMatchedFRS.match_variables
            )
            match_threshold = threshold
            num_groups = group_count
            file_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / f"percentile_matched_{dataset.name}.h5"
            )
            data_format = dataset.data_format
            time_period = dataset.time_period

        return PercentileMatchedFRSFromDataset

    def generate(self):
        from policyengine_uk import Microsimulation

        spi = pd.read_csv(SPI_TAB_FOLDER / "put1920uk.tab", delimiter="\t")
        spi = generate_spi_table(spi)
        simulation = Microsimulation(dataset=self.input_dataset())

        frs = self.input_dataset().load()

        new_values = {}

        for variable in frs.keys():
            if variable not in self.match_variables:
                new_values[variable] = frs[variable][...]
            else:
                targets = simulation.calculate(
                    variable, period=self.input_dataset.time_period
                )
                source_values = spi[variable]
                source_weights = spi.person_weight
                sources = MicroSeries(source_values, weights=source_weights)
                new_values[variable] = match_percentiles(
                    targets,
                    sources,
                    self.match_threshold,
                    self.num_groups,
                )

        self.save_dataset(new_values)
