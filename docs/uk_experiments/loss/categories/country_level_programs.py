from survey_enhance.reweight import LossCategory
from policyengine_core.parameters import ParameterNodeAtInstant
import torch
import numpy as np
from policyengine_core.parameters import ParameterNode, Parameter
from typing import Iterable, Tuple, List
from ..utils import sum_by_household
from survey_enhance.dataset import Dataset


class CountryLevelProgramBudgetaryImpact(LossCategory):
    weight = 1
    static_dataset = False
    variable: str
    taxpayer_only = False

    def get_comparisons(
        self, dataset: Dataset
    ) -> List[Tuple[str, float, torch.Tensor]]:
        countries = dataset.household.country
        pred = []
        targets = []
        names = []

        parameter = self.calibration_parameters.programs._children[
            self.variable
        ]

        if self.taxpayer_only:
            is_taxpayer = dataset.household.income_tax_participants.values
            personal_values = dataset.person[self.variable].values
            personal_values = personal_values[is_taxpayer]
            values = sum_by_household(personal_values, dataset)
        else:
            values = dataset.household[self.variable].values

        # Budgetary impacts

        if "UNITED_KINGDOM" in parameter.budgetary_impact._children:
            pred += [values]
            targets += [parameter.budgetary_impact._children["UNITED_KINGDOM"]]
            names += [f"{self.variable}_budgetary_impact_UNITED_KINGDOM"]

        if "GREAT_BRITAIN" in parameter.budgetary_impact._children:
            pred += [values * (countries != "NORTHERN_IRELAND")]
            targets += [parameter.budgetary_impact._children["GREAT_BRITAIN"]]
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
                names += [f"{self.variable}_budgetary_impact_{single_country}"]

        comparisons = []
        for name, value, target in zip(names, pred, targets):
            comparisons += [(name, value, target)]
        return comparisons


class CountryLevelProgramParticipants(LossCategory):
    weight = 1
    static_dataset = False
    variable: str
    taxpayer_only = False

    def get_comparisons(
        self, dataset: Dataset
    ) -> List[Tuple[str, float, torch.Tensor]]:
        countries = dataset.household.country
        pred = []
        targets = []
        names = []

        parameter = self.calibration_parameters.programs._children[
            self.variable
        ]

        if "participants" in parameter._children:
            if self.taxpayer_only:
                is_taxpayer = dataset.household.income_tax_participants.values
                personal_values = dataset.person[self.variable].values
                personal_values = personal_values[is_taxpayer]
                values = sum_by_household(personal_values > 0, dataset)
            else:
                values = dataset.household[
                    self.variable + "_participants"
                ].values

            if "UNITED_KINGDOM" in parameter.participants._children:
                pred += [values]
                targets += [parameter.participants._children["UNITED_KINGDOM"]]
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
                    names += [f"{self.variable}_participants_{single_country}"]

        comparisons = []
        for name, value, target in zip(names, pred, targets):
            comparisons += [(name, value, target)]
        return comparisons


class CountryLevelProgram(LossCategory):
    weight = None
    static_dataset = False
    variable: str
    taxpayer_only = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parameter = self.calibration_parameters.programs._children[
            self.variable
        ]

        # Budgetary impacts

        if "UNITED_KINGDOM" in parameter.budgetary_impact._children:
            self.weight = parameter.budgetary_impact._children[
                "UNITED_KINGDOM"
            ]
        if (
            self.weight is None
            and "GREAT_BRITAIN" in parameter.budgetary_impact._children
        ):
            self.weight = parameter.budgetary_impact._children["GREAT_BRITAIN"]
        if self.weight is None:
            self.weight = 0
            for single_country in (
                "ENGLAND",
                "WALES",
                "SCOTLAND",
                "NORTHERN_IRELAND",
            ):
                if (
                    single_country in parameter.budgetary_impact._children
                ):
                    self.weight += parameter.budgetary_impact._children[
                        single_country
                    ]

        if self.weight is None or self.weight == 0:
            raise ValueError(
                f"I tried to ensure that {self.variable} is weighted by its budgetary impact, but I couldn't find a budgetary impact for it."
            )

        self.weight /= 1e9

        init_kwargs = {
            "dataset": self.dataset,
            "calibration_parameters": self.calibration_parameters,
            "ancestor": self.ancestor,
            "static_dataset": self.static_dataset,
            "comparison_white_list": self.comparison_white_list,
            "comparison_black_list": self.comparison_black_list,
            "name": self.name,
        }

        budgetary_impact_loss_type = type(
            f"{self.variable}_budgetary_impact",
            (CountryLevelProgramBudgetaryImpact,),
            {
                "variable": self.variable,
                "taxpayer_only": self.taxpayer_only,
            },
        )
        budgetary_impact_loss = budgetary_impact_loss_type(**init_kwargs)

        participant_loss_type = type(
            f"{self.variable}_participants",
            (CountryLevelProgramParticipants,),
            {
                "variable": self.variable,
                "taxpayer_only": self.taxpayer_only,
            },
        )
        participant_loss = participant_loss_type(**init_kwargs)

        self.sublosses = torch.nn.ModuleList(
            [
                budgetary_impact_loss,
                participant_loss,
            ]
        )


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
    taxpayers_only = True


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
    taxpayers_only = True


class SelfEmploymentIncome(CountryLevelProgram):
    variable = "self_employment_income"
    taxpayers_only = True


class PensionIncome(CountryLevelProgram):
    variable = "pension_income"
    taxpayers_only = True


class PropertyIncome(CountryLevelProgram):
    variable = "property_income"
    taxpayers_only = True


class SavingsInterestIncome(CountryLevelProgram):
    variable = "savings_interest_income"
    taxpayers_only = True


class DividendIncome(CountryLevelProgram):
    variable = "dividend_income"
    taxpayers_only = True


class IncomeTax(LossCategory):
    static_dataset = True
    weight = 200

    def get_comparisons(
        self, dataset: Dataset
    ) -> List[Tuple[str, float, torch.Tensor]]:
        income_tax = dataset.person.income_tax
        total_income = dataset.person.adjusted_net_income
        countries = dataset.household.country
        household_income_tax = dataset.household.income_tax

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
                income_tax * income_is_in_band,
                dataset,
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

        tax_band = dataset.person.tax_band
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

        person_country = dataset.person.country

        for country in ("ENGLAND", "WALES", "NORTHERN_IRELAND", "SCOTLAND"):
            for band in ("BASIC", "HIGHER", "ADDITIONAL"):
                comparisons += [
                    (
                        f"income_tax_payers_{country}_{band}",
                        sum_by_household(
                            (income_tax > 0)
                            * (person_country == country)
                            * (tax_band == band),
                            dataset,
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
