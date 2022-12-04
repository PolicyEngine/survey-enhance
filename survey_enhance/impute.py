import pandas as pd
from typing import List


def extend_survey_table(
    df: pd.DataFrame,
    id_columns: List[str],
    weight_column: str = None,
    weight_multiplier: float = 0.0,
    num_duplicates: int = 1,
) -> pd.DataFrame:
    """
    Extend the survey table by duplicating its entities.

    Args:
        df: The survey table.
        id_columns: The columns that uniquely identify an entity. These columns will not be copied exactly, but will be suffixed with a number.
        weight_column: The column that contains the weight of an entity. This column will be multiplied by the weight multiplier.
        weight_multiplier: The multiplier for the weight column.
        num_duplicates: The number of times each entity will be duplicated.

    Returns:
        The extended survey table.
    """
    extended_df = df.copy()
    extended_df["clone_id"] = 0
    for i in range(num_duplicates):
        clone_df = df.copy()
        clone_df["clone_id"] = i + 1
        for id_column in id_columns:
            clone_df[id_column] = 100 * df[id_column].values + i + 1
        if weight_column is not None:
            clone_df[weight_column] = (
                df[weight_column].values * weight_multiplier
            )
        extended_df = pd.concat([extended_df, clone_df])
    return extended_df.reset_index(drop=True)
