from survey_enhance.loss import LossCategory, Dataset
from survey_enhance.dataset import sum_by_household
from policyengine_core.parameters import ParameterNodeAtInstant
import torch
import numpy as np
from policyengine_core.parameters import ParameterNode, Parameter
from typing import Iterable, Tuple, List

class CountryLevelProgramBudgetaryImpact(LossCategory):
    weight = 1
    static_dataset = False
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

        comparisons = []
        for name, value, target in zip(names, pred, targets):
            comparisons += [(name, value, target)]
        return comparisons

class CountryLevelProgramParticipants(LossCategory):
    weight = 1
    static_dataset = False
    variable: str

    def get_comparisons(self, dataset: Dataset) -> List[Tuple[str, float, torch.Tensor]]:
        countries = dataset.household_df.country
        pred = []
        targets = []
        names = []

        parameter = self.calibration_parameters.programs._children[
            self.variable
        ]

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

class CountryLevelProgram(LossCategory):
    weight = None
    static_dataset = False
    variable: str

    def __init__(self, dataset: Dataset, calibration_parameters: ParameterNodeAtInstant, weight: float = None, ancestor: "LossCategory" = None):
        super().__init__(dataset, calibration_parameters, weight, ancestor)
        parameter = self.calibration_parameters.programs._children[
            self.variable
        ]

        # Budgetary impacts

        if "UNITED_KINGDOM" in parameter.budgetary_impact._children:
            self.weight = parameter.budgetary_impact._children["UNITED_KINGDOM"]
        if self.weight is None and "GREAT_BRITAIN" in parameter.budgetary_impact._children:
            self.weight = parameter.budgetary_impact._children["GREAT_BRITAIN"]

        for single_country in (
            "ENGLAND",
            "WALES",
            "SCOTLAND",
            "NORTHERN_IRELAND",
        ):
            if self.weight is None and single_country in parameter.budgetary_impact._children:
                self.weight = parameter.budgetary_impact._children[single_country]

        if self.weight is None:
            raise ValueError(f"I tried to ensure that {self.variable} is weighted by its budgetary impact, but I couldn't find a budgetary impact for it.")

        budgetary_impact_loss = type(
            f"{self.variable}_budgetary_impact",
            (CountryLevelProgramBudgetaryImpact,),
            {"variable": self.variable, "ancestor": self.ancestor},
        )

        participant_loss = type(
            f"{self.variable}_participants",
            (CountryLevelProgramParticipants,),
            {"variable": self.variable, "ancestor": self.ancestor},
        )
        
        self.sublosses = torch.nn.ModuleList([
            subcategory(dataset, self.calibration_parameters)
            for subcategory in (
                budgetary_impact_loss,
                participant_loss,
            )
        ])


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

class IncomeTax(LossCategory):
    weight = 1
    static_dataset = False

    def get_comparisons(self, dataset: Dataset) -> List[Tuple[str, float, torch.Tensor]]:
        income_tax = dataset.person_df.income_tax
        total_income = dataset.person_df.adjusted_net_income
        countries = dataset.household_df.country
        household_income_tax = dataset.household_df.income_tax

        it = self.calibration_parameters.programs.income_tax

        comparisons = []

        # Revenue by country

        for country in ("ENGLAND", "WALES", "NORTHERN_IRELAND", "SCOTLAND"):
            comparisons += [
                (
                    f"income_tax_{country}",
                    household_income_tax * (countries == country),
                    it.budgetary_impact.by_country._children[country],
                )
            ]

        comparisons += [
            (
                "income_tax_UNITED_KINGDOM",
                household_income_tax,
                it.budgetary_impact.by_country._children["UNITED_KINGDOM"],
            )
        ]

        # Revenue by income band

        scale = it.budgetary_impact.by_income

        for i in range(len(scale.thresholds)):
            lower_threshold = scale.thresholds[i]
            upper_threshold = (
                scale.thresholds[i + 1]
                if i < len(scale.thresholds) - 1
                else np.inf
            )

            income_is_in_band = (total_income >= lower_threshold) * (
                total_income < upper_threshold
            )
            household_values = sum_by_household(
                income_tax * income_is_in_band, dataset,
            )

            amount = scale.amounts[i]
            comparisons += [
                (
                    f"income_tax_by_income_{i}",
                    household_values,
                    amount,
                )
            ]

        # Taxpayers by country and income band

        tax_band = dataset.person_df.tax_band
        tax_band = np.select(
            [
                tax_band == "STARTER",
                tax_band == "INTERMEDIATE",
                True,
            ],
            [
                "BASIC",
                "BASIC",
                tax_band,
            ],
        )

        person_country = dataset.household_df.country

        for country in ("ENGLAND", "WALES", "NORTHERN_IRELAND", "SCOTLAND"):
            for band in ("BASIC", "HIGHER", "ADDITIONAL"):
                comparisons += [
                    (
                        f"income_tax_payers_{country}_{band}",
                        sum_by_household(
                            (income_tax > 0)
                            * (person_country == country)
                            * (tax_band == band),
                            dataset
                        ),
                        it.participants.by_country_and_band._children[
                            country
                        ]._children[band],
                    )
                ]
        
        return comparisons

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
    IncomeTax,
]