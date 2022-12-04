from survey_enhance.experiment.initialisation import (
    dataset,
    calibration_parameters,
    Loss,
    create_frs_dataset,
    household_weights,
)
from survey_enhance.reweight import LossCategory
from survey_enhance.experiment.loss.loss import Programs
from typing import Type

loss = Loss(dataset, calibration_parameters, static_dataset=True)

initial_loss = loss(household_weights, dataset)

NUM_HOLDOUT_SETS = 10
holdout_sets = loss.create_holdout_sets(
    dataset, NUM_HOLDOUT_SETS, exclude_by_name="Demographics"
)

import torch
from torch import optim
import numpy as np
import pandas as pd

torch.autograd.set_grad_enabled(True)

training_df = pd.DataFrame()

for i in range(1):
    weight_adjustment = torch.tensor(
        np.zeros(household_weights.shape), requires_grad=True
    )
    optimiser = optim.Adam([weight_adjustment], lr=1e0)
    validation_metric_names = holdout_sets[i]
    training_loss_fn = Loss(
        dataset,
        calibration_parameters,
        static_dataset=True,
        comparison_black_list=validation_metric_names,
    )
    validation_loss_fn = Loss(
        dataset,
        calibration_parameters,
        static_dataset=True,
        comparison_white_list=validation_metric_names,
    )

    for j in range(10_000):
        optimiser.zero_grad()
        # Apply ReLU to weights + adjustment
        adjusted_weights = torch.nn.functional.relu(
            household_weights + weight_adjustment
        )
        loss_value: torch.Tensor = training_loss_fn(adjusted_weights, dataset)
        validation_loss_value: torch.Tensor = validation_loss_fn(
            adjusted_weights, dataset
        )
        average_loss_value = (
            NUM_HOLDOUT_SETS * loss_value.item() - validation_loss_value.item()
        ) / (NUM_HOLDOUT_SETS - 1)
        loss_value.backward(retain_graph=True)
        optimiser.step()
        if j % 10 == 0:
            print(
                f"Holdout set {i}, Epoch {j}: \tLoss {loss_value.item() - 1:+.3%}, Validation loss {validation_loss_value.item() - 1:+.3%}, Average loss {average_loss_value - 1:+.3%}"
            )

    set_df = training_loss_fn.collect_comparison_log()
    set_validation_df = validation_loss_fn.collect_comparison_log()
    set_df["holdout_set"] = i
    set_validation_df["holdout_set"] = i
    set_df["validation"] = False
    set_validation_df["validation"] = True
    training_df = pd.concat([training_df, set_df, set_validation_df])
    training_df.to_csv("training_log.csv")
