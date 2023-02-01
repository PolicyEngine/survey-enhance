from survey_enhance.dataset import Dataset
from pathlib import Path
import pandas as pd
from datasets.frs.imputations.income import SPI_TAB_FOLDER, generate_spi_table
from datasets.frs.frs import FRS_2019_20
from survey_enhance.percentile_match import match_percentiles
from typing import List, Type


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
    input_dataset = FRS_2019_20

    @staticmethod
    def from_dataset(
        dataset: Type[Dataset],
        percentile_matched_variables: List[str] = None,
        threshold: float = 0.95,
        group_count: int = 10,
        force_generate: bool = False,
        force_not_generate: bool = False,
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

        dataset = PercentileMatchedFRSFromDataset()
        if not force_not_generate and (force_generate or not dataset.exists):
            dataset.generate()

        return PercentileMatchedFRSFromDataset

    def generate(self):
        spi = pd.read_csv(SPI_TAB_FOLDER / "put1920uk.tab", delimiter="\t")
        spi = generate_spi_table(spi)

        frs = self.input_dataset().load()

        new_values = {}

        for variable in frs.keys():
            if variable not in self.match_variables:
                new_values[variable] = frs[variable][...]
            else:
                new_values[variable] = match_percentiles(
                    frs[variable][...],
                    spi[variable],
                    self.match_threshold,
                    self.num_groups,
                )

        self.save_dataset(new_values)
