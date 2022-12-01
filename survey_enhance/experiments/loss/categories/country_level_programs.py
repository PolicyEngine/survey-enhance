from survey_enhance.loss import LossCategory, Dataset
from policyengine_uk.tools.simulation import Microsimulation
import torch
import numpy as np
from policyengine_core.parameters import ParameterNode, Parameter
from typing import Iterable, Tuple, List


class CountryLevelProgram(LossCategory):
    weight = 1
    static_dataset = True
    variable: str

    def get_comparisons(self, dataset: Dataset) -> List[Tuple[str, float, torch.Tensor]]:
        countries = dataset.household_df.country
        pred = []
        targets = []
        names = []

        parameter = self.calibration_parameters.programs._children[
            self.variable
        ]

        values = dataset.household_df[self.variable].values

        # Budgetary impacts

        if "UNITED_KINGDOM" in parameter.budgetary_impact._children:
            pred += [values]
            targets += [
                parameter.budgetary_impact._children["UNITED_KINGDOM"]
            ]
            names += [f"{self.variable}_budgetary_impact_UNITED_KINGDOM"]
        if "GREAT_BRITAIN" in parameter.budgetary_impact._children:
            pred += [values * (countries != "NORTHERN_IRELAND")]
            targets += [
                parameter.budgetary_impact._children["GREAT_BRITAIN"]
            ]
            names += [f"{self.variable}_budgetary_impact_GREAT_BRITAIN"]

        for single_country in (
            "ENGLAND",
            "WALES",
            "SCOTLAND",
            "NORTHERN_IRELAND",
        ):
            if single_country in parameter.budgetary_impact._children:
                pred += [values * (countries == single_country)]
                targets += [
                    parameter.budgetary_impact._children[single_country]
                ]
                names += [
                    f"{self.variable}_budgetary_impact_{single_country}"
                ]

        # Participants

        if "participants" in parameter._children:
            values = dataset.household_df[f"{self.variable}_participants"]

            if "UNITED_KINGDOM" in parameter.participants._children:
                pred += [values]
                targets += [
                    parameter.participants._children["UNITED_KINGDOM"]
                ]
                names += [f"{self.variable}_participants_UNITED_KINGDOM"]
            if "GREAT_BRITAIN" in parameter.participants._children:
                pred += [values * (countries != "NORTHERN_IRELAND")]
                targets += [parameter.participants._children["GREAT_BRITAIN"]]
                names += [f"{self.variable}_participants_GREAT_BRITAIN"]

            for single_country in (
                "ENGLAND",
                "WALES",
                "SCOTLAND",
                "NORTHERN_IRELAND",
            ):
                if single_country in parameter.participants._children:
                    pred += [values * (countries == single_country)]
                    targets += [
                        parameter.participants._children[single_country]
                    ]
                    names += [
                        f"{self.variable}_participants_{single_country}"
                    ]
        
        comparisons = []
        for name, value, target in zip(names, pred, targets):
            comparisons += [(name, value, target)]
        return comparisons


class IncomeSupport(CountryLevelProgram):
    variable = "income_support"


class PensionCredit(CountryLevelProgram):
    variable = "pension_credit"


class WorkingTaxCredit(CountryLevelProgram):
    variable = "working_tax_credit"


class ChildBenefit(CountryLevelProgram):
    variable = "child_benefit"


class ChildTaxCredit(CountryLevelProgram):
    variable = "child_tax_credit"


class UniversalCredit(CountryLevelProgram):
    variable = "universal_credit"


class StatePension(CountryLevelProgram):
    variable = "state_pension"


class TotalNI(CountryLevelProgram):
    variable = "total_NI"


class JSAIncome(CountryLevelProgram):
    variable = "JSA_income"


class CouncilTax(CountryLevelProgram):
    variable = "council_tax_less_benefit"


class HousingBenefit(CountryLevelProgram):
    variable = "housing_benefit"


class ESAIncome(CountryLevelProgram):
    variable = "ESA_income"


class EmploymentIncome(CountryLevelProgram):
    variable = "employment_income"


class SelfEmploymentIncome(CountryLevelProgram):
    variable = "self_employment_income"


class PensionIncome(CountryLevelProgram):
    variable = "pension_income"


class PropertyIncome(CountryLevelProgram):
    variable = "property_income"


class SavingsInterestIncome(CountryLevelProgram):
    variable = "savings_interest_income"


class DividendIncome(CountryLevelProgram):
    variable = "dividend_income"

country_level_programs = [
    UniversalCredit,
    ChildBenefit,
    ChildTaxCredit,
    WorkingTaxCredit,
    PensionCredit,
    IncomeSupport,
    StatePension,
    HousingBenefit,
    ESAIncome,
    JSAIncome,
    CouncilTax,
    TotalNI,
    EmploymentIncome,
    SelfEmploymentIncome,
    PensionIncome,
    SavingsInterestIncome,
    PropertyIncome,
    DividendIncome,
]