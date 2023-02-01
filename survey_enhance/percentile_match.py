import pandas as pd
import numpy as np
from microdf import MicroDataFrame, MicroSeries
from typing import Union

DataFrame = Union[pd.DataFrame, MicroDataFrame]
Series = Union[pd.Series, MicroSeries]


def match_percentiles_df(
    target_df: DataFrame,
    source_df: DataFrame,
    percentile_threshold: float = 0.95,
    num_groups: int = 10,
) -> pd.DataFrame:
    """
    Match the percentiles of the source_df to the target_df.

    Args:
        target_df: The DataFrame to edit to match the source_df's percentiles.
        source_df: The DataFrame to match the percentiles to.
        percentile_threshold: Don't adjust data for percentiles below this threshold.
        num_groups: The number of percentile groups to split the data into.

    Returns:
        A DataFrame with the same index as target_df, but with the adjusted values.
    """

    target_df = target_df.copy()

    for column in target_df.columns:
        target_df[column] = match_percentiles(
            target_df[column],
            source_df[column],
            percentile_threshold,
            num_groups,
        )

    return target_df


def match_percentiles(
    targets: Series,
    sources: Series,
    percentile_threshold: float = 0.95,
    num_groups: int = 10,
) -> pd.Series:
    """
    Match the percentiles of the source Series to the target Series.

    Args:
        targets: The Series to edit to match the source Series's percentiles.
        sources: The Series to match the percentiles to.
        percentile_threshold: Don't adjust data for percentiles below this threshold.
        num_groups: The number of percentile groups to split the data into.

    Returns:
        A Series with the same index as target_df, but with the adjusted values.
    """
    if not isinstance(targets, MicroSeries):
        targets = MicroSeries(targets)
    if not isinstance(sources, MicroSeries):
        sources = MicroSeries(sources)
    targets = targets.copy()

    percentile_boundaries = np.linspace(
        percentile_threshold, 1, num_groups + 1
    )
    lower_percentiles = percentile_boundaries[:-1]
    upper_percentiles = percentile_boundaries[1:]

    for lower, upper in zip(lower_percentiles, upper_percentiles):
        lower_target = targets[targets > 0].quantile(lower)
        upper_target = targets[targets > 0].quantile(upper)
        lower_source = sources[sources > 0].quantile(lower)
        upper_source = sources[sources > 0].quantile(upper)

        # Replace all values in the target Series that fall within the current
        # percentile range with the mean of the source Series's values in the
        # same percentile range.

        target_in_range = (targets >= lower_target) & (targets <= upper_target)
        source_in_range = (sources >= lower_source) & (sources <= upper_source)

        targets[target_in_range] = sources[source_in_range].mean()
    return targets
