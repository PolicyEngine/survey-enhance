import numpy as np
import pandas as pd
from survey_enhance.survey import Survey


def sum_by_household(values: pd.Series, dataset: Survey) -> np.ndarray:
    return (
        pd.Series(values)
        .groupby(dataset.person.person_household_id.values)
        .sum()
        .values
    )
