from survey_enhance.experiment.dataset import (
    dataset,
    calibration_parameters,
    create_frs_dataset,
    Dataset,
)
from survey_enhance.impute import extend_survey_table
from survey_enhance.experiment.loss.loss import Loss
import torch

extended_person_df = extend_survey_table(
    dataset.person_df,
    ["person_id", "person_benunit_id", "person_household_id"],
    None,
    0.0,
    1,
)

extended_benunit_df = extend_survey_table(
    dataset.benunit_df,
    ["benunit_id", "household_id"],
    None,
    0.0,
    1,
)

extended_household_df = extend_survey_table(
    dataset.household_df,
    ["household_id"],
    "household_weight",
    0.0,
    1,
)

extended_dataset = Dataset(
    extended_person_df, extended_benunit_df, extended_household_df
)

extended_person_df.to_csv("extended_person_df.csv")
extended_benunit_df.to_csv("extended_benunit_df.csv")
extended_household_df.to_csv("extended_household_df.csv")

household_weights = torch.tensor(
    extended_dataset.household_df.household_weight.values
)
loss = Loss(extended_dataset, calibration_parameters)

loss_value = loss(household_weights, extended_dataset)

ldf_1 = loss.collect_comparison_log()

original_loss = Loss(dataset, calibration_parameters)
original_household_weights = torch.tensor(
    dataset.household_df.household_weight.values
)

original_loss_value = original_loss(original_household_weights, dataset)

ldf_2 = original_loss.collect_comparison_log()

total_loss_1 = ldf_1[ldf_1.epoch == 1].loss.sum()
total_loss_2 = ldf_2[ldf_2.epoch == 1].loss.sum()

print(total_loss_1, total_loss_2)
