from survey_enhance.survey import Survey
from pathlib import Path
import pandas as pd
from pandas import DataFrame
import warnings
from ..utils import (
    sum_to_entity,
    categorical,
    sum_from_positive_fields,
    sum_positive_variables,
    fill_with_mean,
    concatenate_surveys,
)
from typing import Dict
import numpy as np
from numpy import maximum as max_, where
from typing import Type


class RawFRS(Survey):
    """A `Survey` instance for the Family Resources Survey."""

    name = "raw_frs"
    label = "Family Resources Survey"

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
                ).apply(pd.to_numeric, errors="coerce")

            sernum = (
                "sernum"
                if "sernum" in tables[table_name].columns
                else "SERNUM"
            )  # FRS inconsistently users sernum/SERNUM in different years

            if "PERSON" in tables[table_name].columns:
                tables[table_name]["person_id"] = (
                    tables[table_name][sernum] * 1e2
                    + tables[table_name].BENUNIT * 1e1
                    + tables[table_name].PERSON
                ).astype(int)

            if "BENUNIT" in tables[table_name].columns:
                tables[table_name]["benunit_id"] = (
                    tables[table_name][sernum] * 1e2
                    + tables[table_name].BENUNIT * 1e1
                ).astype(int)

            if sernum in tables[table_name].columns:
                tables[table_name]["household_id"] = (
                    tables[table_name][sernum] * 1e2
                ).astype(int)
            if table_name in ("adult", "child"):
                tables[table_name].set_index(
                    "person_id", inplace=True, drop=False
                )
            elif table_name == "benunit":
                tables[table_name].set_index(
                    "benunit_id", inplace=True, drop=False
                )
            elif table_name == "househol":
                tables[table_name].set_index(
                    "household_id", inplace=True, drop=False
                )
        tables["benunit"] = tables["benunit"][
            tables["benunit"].benunit_id.isin(tables["adult"].benunit_id)
        ]
        tables["househol"] = tables["househol"][
            tables["househol"].household_id.isin(tables["adult"].household_id)
        ]

        # Save the data
        self.save(tables)


class RawFRS_2018_19(RawFRS):
    name = "raw_frs_2018_19"
    label = "Family Resources Survey 2018-19 (raw)"


class RawFRS_2019_20(RawFRS):
    name = "raw_frs_2019_20"
    label = "Family Resources Survey 2019-20 (raw)"


class RawFRS_2020_21(RawFRS):
    name = "raw_frs_2020_21"
    label = "Family Resources Survey 2020-21 (raw)"
