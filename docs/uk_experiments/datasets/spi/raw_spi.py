from survey_enhance.survey import Survey
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
from typing import Type, Dict


class RawSPI(Survey):
    """A `Survey` instance for the Family Resources Survey."""

    name = "raw_spi"
    label = "Survey of Personal Incomes"

    main_table_name = "put1920uk"

    def generate(self, tab_folder: Path):
        """Generate the survey data from the original TAB files.

        Args:
            tab_folder (Path): The folder containing the original TAB files.
        """

        if isinstance(tab_folder, str):
            tab_folder = Path(tab_folder)

        # Load the data
        tables = {}
        tab_file = tab_folder / f"{self.main_table_name}.tab"
        table_name = "main"
        tables[table_name] = pd.read_csv(
            tab_file,
            sep="\t",
            low_memory=False,
        ).apply(pd.to_numeric, errors="coerce")

        # Save the data
        self.save(tables)


class RawSPI_2019_20(RawSPI):
    name = "raw_spi_2019_20"
    label = "Survey of Personal Incomes 2019-20 (raw)"

    main_table_name = "put1920uk"
