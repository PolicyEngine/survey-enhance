import pandas as pd
from typing import List, Dict
import numpy as np
from policyengine_core.data import Dataset
import pickle


def sum_to_entity(
    values: pd.Series, foreign_key: pd.Series, primary_key
) -> pd.Series:
    """Sums values by joining foreign and primary keys.

    Args:
        values (pd.Series): The values in the non-entity table.
        foreign_key (pd.Series): E.g. pension.person_id.
        primary_key ([type]): E.g. person.index.

    Returns:
        pd.Series: A value for each person.
    """
    return values.groupby(foreign_key).sum().reindex(primary_key).fillna(0)


def categorical(
    values: pd.Series, default: int, left: list, right: list
) -> pd.Series:
    """Maps a categorical input to an output using given left and right arrays.

    Args:
        values (pd.Series): The input values.
        default (int): A default value (to replace NaNs).
        left (list): The left side of the map.
        right (list): The right side of the map.

    Returns:
        pd.Series: The mapped values.
    """
    return values.fillna(default).map({i: j for i, j in zip(left, right)})


def sum_from_positive_fields(
    table: pd.DataFrame, fields: List[str]
) -> np.array:
    """Sum from fields in table, ignoring negative values.

    Args:
        table (DataFrame)
        fields (List[str])

    Returns:
        np.array
    """
    return np.where(
        table[fields].sum(axis=1) > 0, table[fields].sum(axis=1), 0
    )


def sum_positive_variables(variables: List[str]) -> np.array:
    """Sum positive variables.

    Args:
        variables (List[str])

    Returns:
        np.array
    """
    return sum([np.where(variable > 0, variable, 0) for variable in variables])


def fill_with_mean(
    table: pd.DataFrame, code: str, amount: str, multiplier: float = 52
) -> np.array:
    """Fills missing values in a table with the mean of the column.

    Args:
        table (DataFrame): Table to fill.
        code (str): Column signifying existence.
        amount (str): Column with values.
        multiplier (float): Multiplier to apply to amount.

    Returns:
        np.array: Filled values.
    """
    needs_fill = (table[code] == 1) & (table[amount] < 0)
    has_value = (table[code] == 1) & (table[amount] >= 0)
    fill_mean = table[amount][has_value].mean()
    filled_values = np.where(needs_fill, fill_mean, table[amount])
    return np.maximum(filled_values, 0) * multiplier


def concatenate_two_surveys(
    primary_tables: Dict[str, pd.DataFrame],
    secondary_tables: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Concatenate a survey with another, by adjusting ID variable values.

    Args:
        primary_tables (Dict[str, pd.DataFrame]): The first survey.
        secondary_tables (Dict[str, pd.DataFrame]): The second survey.
    """

    new_tables = {}

    for table_name, table in primary_tables.items():
        if table_name in secondary_tables:
            other_table = secondary_tables[table_name]
            id_vars = [col for col in table.columns if col.endswith("_id")]
            for id_var in id_vars:
                other_table[id_var] = (
                    other_table[id_var] + table[id_var].max() + 1
                )
            # Update index
            other_table.index = other_table.index + table.index.max() + 1
            new_tables[table_name] = pd.concat([table, other_table])
        else:
            new_tables[table_name] = table

    return new_tables


def concatenate_surveys(surveys: List[Dataset]) -> Dataset:
    """
    Concatenate a list of surveys.

    Args:
        surveys (List[Survey]): A list of surveys.
    """
    new_survey = surveys[0]
    for survey in surveys[1:]:
        new_survey = concatenate_two_surveys(new_survey, survey)
    return new_survey
