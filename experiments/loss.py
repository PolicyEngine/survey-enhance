import torch
import pandas as pd

def uk_household_survey_loss(
    person_df: pd.DataFrame,
    household_df: pd.DataFrame,
    benunit_df: pd.DataFrame,
    person_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Loss function for UK household survey data.
    """