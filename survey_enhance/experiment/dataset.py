import pandas as pd
import numpy as np
from survey_enhance.dataset import Dataset as GenericDataset


class Dataset(GenericDataset):
    person_df: pd.DataFrame
    benunit_df: pd.DataFrame
    household_df: pd.DataFrame

    def __init__(
        self,
        person_df: pd.DataFrame,
        benunit_df: pd.DataFrame,
        household_df: pd.DataFrame,
    ):
        self.person_df = person_df
        self.benunit_df = benunit_df
        self.household_df = household_df


def sum_by_household(values: pd.Series, dataset: Dataset) -> np.ndarray:
    return (
        pd.Series(values)
        .groupby(dataset.person_df.person_household_id)
        .sum()
        .values
    )
