from survey_enhance.survey import Survey
from survey_enhance.reweight import CalibratedWeights
from loss.loss import Loss, calibration_parameters
import torch
import numpy as np  


class CalibratedFRS_2020_21_22(Survey):
    name = "calibrated_frs_2020_21_22"
    label = "Calibrated FRS 2020-21 (over 22)"

    def generate(self):
        from .frs import FRS_2020_OUT_22

        frs_2020_out_22 = FRS_2020_OUT_22()

        original_weights = frs_2020_out_22.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            frs_2020_out_22,
            Loss,
            calibration_parameters,
        )

        weights = calibrated_weights.calibrate(
            "2022-01-01",
            epochs=100,
            learning_rate=0.1,
        )

        tables = dict(
            person=frs_2020_out_22.person.copy(),
            benunit=frs_2020_out_22.benunit.copy(),
            household=frs_2020_out_22.household.copy(),
        )

        tables["household"]["household_weight"] = weights

        self.save(tables)



class CalibratedFRS_2018_21_22(Survey):
    name = "calibrated_frs_2018_21_22"
    label = "Calibrated FRS 2018-21 (over 22)"

    def generate(self):
        from .frs import FRS_2018_21_OUT_22

        frs_2020_out_22 = FRS_2018_21_OUT_22()

        original_weights = frs_2020_out_22.household.household_weight.values

        calibrated_weights = CalibratedWeights(
            original_weights,
            frs_2020_out_22,
            Loss,
            calibration_parameters,
        )

        weights = calibrated_weights.calibrate(
            "2022-01-01",
            epochs=10_000,
            learning_rate=1e-2,
        )

        tables = dict(
            person=frs_2020_out_22.person.copy(),
            benunit=frs_2020_out_22.benunit.copy(),
            household=frs_2020_out_22.household.copy(),
        )

        tables["household"]["household_weight"] = weights

        self.save(tables)