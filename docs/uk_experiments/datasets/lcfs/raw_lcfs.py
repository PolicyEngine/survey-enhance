from survey_enhance.survey import Survey
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
from typing import Type, Dict
import warnings


class RawLCFS(Survey):
    name = "raw_lcfs"
    label = "Living Costs and Food Survey"

    def generate(self, tab_folder: Path):
        """Generate the survey data from the original TAB files.

        Args:
            tab_folder (Path): The folder containing the original TAB files.
        """

        if isinstance(tab_folder, str):
            tab_folder = Path(tab_folder)

        # Load the data
        tables = {}
        for tab_file in tab_folder.glob("*.tab"):
            table_name = tab_file.stem
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables[table_name] = pd.read_csv(
                    tab_file, delimiter="\t", low_memory=False
                ).apply(pd.to_numeric, errors="ignore")

        # Save the data
        self.save(tables)


class RawLCFS_2020_21(RawLCFS):
    name = "raw_lcfs_2020_21"
    label = "Living Costs and Food Survey 2020-21 (raw)"
