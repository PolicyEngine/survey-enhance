from survey_enhance.survey import Survey
import pandas as pd
from pandas import DataFrame
from ..utils import (
    sum_to_entity,
    categorical,
    sum_from_positive_fields,
    sum_positive_variables,
    fill_with_mean,
    concatenate_surveys,
)
from typing import Dict
import numpy as np
from numpy import maximum as max_, where
from typing import Type
from .raw_frs import RawFRS_2018_19, RawFRS_2019_20, RawFRS_2020_21, RawFRS


class FRS(Survey):
    name = "frs"
    label = "Family Resources Survey"

    raw_frs: Type[RawFRS]

    def generate(self):
        raw_frs = self.raw_frs()

        TABLES = (
            "adult",
            "child",
            "accounts",
            "benefits",
            "job",
            "oddjob",
            "benunit",
            "househol",
            "chldcare",
            "pension",
            "maint",
            "mortgage",
            "penprov",
        )
        (
            adult,
            child,
            accounts,
            benefits,
            job,
            oddjob,
            benunit,
            household,
            childcare,
            pension,
            maintenance,
            mortgage,
            pen_prov,
        ) = [raw_frs.load(table) for table in TABLES]

        person = pd.concat([adult, child]).sort_index().fillna(0)

        tables = dict(
            person=pd.DataFrame(index=person.index),
            benunit=pd.DataFrame(index=benunit.index),
            household=pd.DataFrame(index=household.index),
        )

        tables = add_id_variables(tables, person, benunit, household)
        tables = add_personal_variables(tables, person)
        tables = add_benunit_variables(tables, benunit)
        tables = add_household_variables(tables, household)
        tables = add_market_income(
            tables, person, pension, accounts, household, oddjob
        )
        tables = add_benefit_income(tables, person, benefits, household)
        tables = add_expenses(
            tables,
            person,
            job,
            household,
            maintenance,
            mortgage,
            childcare,
            pen_prov,
        )
        self.save(tables)


class FRS_2018_19(FRS):
    name = "frs_2018_19"
    label = "Family Resources Survey 2018-19"
    raw_frs = RawFRS_2018_19


class FRS_2019_20(FRS):
    name = "frs_2019_20"
    label = "Family Resources Survey 2019-20"
    raw_frs = RawFRS_2019_20


class FRS_2020_21(FRS):
    name = "frs_2020_21"
    label = "Family Resources Survey 2020-21"
    raw_frs = RawFRS_2020_21


def add_id_variables(
    tables: Dict[str, DataFrame],
    person: DataFrame,
    benunit: DataFrame,
    household: DataFrame,
):
    """Adds ID variables and weights.

    Args:
        tables (Dict[str, DataFrame])
        person (DataFrame)
        benunit (DataFrame)
        household (DataFrame)
    """
    # Add primary and foreign keys
    tables["person"]["person_id"] = person.index
    tables["person"]["person_benunit_id"] = person.benunit_id
    tables["person"]["person_household_id"] = person.household_id
    tables["benunit"]["benunit_id"] = benunit.benunit_id
    tables["household"]["household_id"] = household.household_id

    # Add grossing weights
    tables["household"]["household_weight"] = household.GROSS4
    return tables


def add_personal_variables(tables: Dict[str, DataFrame], person: DataFrame):
    """Adds personal variables (age, gender, education).

    Args:
        tables (Dict[str, DataFrame])
        person (DataFrame)
    """
    # Add basic personal variables
    age = person.AGE80.fillna(0) + person.AGE.fillna(0)
    tables["person"]["age"] = age
    # Age fields are AGE80 (top-coded) and AGE in the adult and child tables, respectively.
    tables["person"]["gender"] = np.where(person.SEX == 1, "MALE", "FEMALE")
    tables["person"]["hours_worked"] = person.TOTHOURS * 52
    tables["person"]["is_household_head"] = person.HRPID == 1
    tables["person"]["is_benunit_head"] = person.UPERSON == 1
    MARITAL = [
        "MARRIED",
        "SINGLE",
        "SINGLE",
        "WIDOWED",
        "SEPARATED",
        "DIVORCED",
    ]
    tables["person"]["marital_status"] = categorical(
        person.MARITAL.replace(0, 2), 2, range(1, 7), MARITAL
    )

    # Add education levels
    fted = person.FTED.astype(int)
    typeed2 = person.TYPEED2.astype(int)
    not_in_education = fted.isin((2, -1, 0))
    pre_primary = typeed2 == 1
    primary = (
        typeed2.isin((2, 4))
        | (typeed2.isin((3, 8)) & (age < 11))
        | ((typeed2 == 0) & (fted == 1) & (age > 5) & (age < 11))
    )
    lower_secondary = (
        typeed2.isin((5, 6))
        | (typeed2.isin((3, 8)) & (age >= 11) & (age <= 16))
        | ((typeed2 == 0) & (fted == 1) & (age <= 16))
    )
    non_advanced_further_education = typeed2 == 7 | (
        typeed2.isin((3, 8)) & (age > 16)
    ) | ((typeed2 == 0) & (fted == 1) & (age > 16))
    upper_secondary = typeed2.isin((7, 8)) & (age < 19)
    post_secondary = (
        typeed2.isin((6,)) & (age >= 18) | non_advanced_further_education
    )
    higher_education = typeed2.isin((9,))

    labels = [
        "NOT_IN_EDUCATION",
        "PRE_PRIMARY",
        "PRIMARY",
        "LOWER_SECONDARY",
        "UPPER_SECONDARY",
        "POST_SECONDARY",
        "TERTIARY",
    ]

    education = np.select(
        [
            not_in_education,
            pre_primary,
            primary,
            lower_secondary,
            upper_secondary,
            post_secondary,
            higher_education,
        ],
        labels,
        default="NOT_IN_EDUCATION",
    )

    tables["person"]["current_education"] = education

    # Add employment status
    EMPLOYMENTS = [
        "CHILD",
        "FT_EMPLOYED",
        "PT_EMPLOYED",
        "FT_SELF_EMPLOYED",
        "PT_SELF_EMPLOYED",
        "UNEMPLOYED",
        "RETIRED",
        "STUDENT",
        "CARER",
        "LONG_TERM_DISABLED",
        "SHORT_TERM_DISABLED",
        "OTHER",
    ]
    tables["person"]["employment_status"] = categorical(
        person.EMPSTATI, 1, range(12), EMPLOYMENTS
    )
    return tables


def add_household_variables(
    tables: Dict[str, DataFrame], household: DataFrame
):
    """Adds household variables (region, tenure, council tax imputation).

    Args:
        tables (Dict[str, DataFrame])
        household (DataFrame)
    """
    # Add region
    from policyengine_uk.variables.household.demographic.household import (
        Region,
    )

    REGIONS = [
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
        "UNKNOWN",
    ]
    tables["household"]["region"] = categorical(
        household.GVTREGNO, 14, [1, 2] + list(range(4, 15)), REGIONS
    )
    TENURES = [
        "RENT_FROM_COUNCIL",
        "RENT_FROM_HA",
        "RENT_PRIVATELY",
        "RENT_PRIVATELY",
        "OWNED_OUTRIGHT",
        "OWNED_WITH_MORTGAGE",
    ]
    tables["household"]["tenure_type"] = categorical(
        household.PTENTYP2, 3, range(1, 7), TENURES
    )
    tables["household"]["num_bedrooms"] = household.BEDROOM6
    ACCOMMODATIONS = [
        "HOUSE_DETACHED",
        "HOUSE_SEMI_DETACHED",
        "HOUSE_TERRACED",
        "FLAT",
        "CONVERTED_HOUSE",
        "MOBILE",
        "OTHER",
    ]
    tables["household"]["accommodation_type"] = categorical(
        household.TYPEACC, 1, range(1, 8), ACCOMMODATIONS
    )

    # Impute Council Tax

    # Only ~25% of household report Council Tax bills - use
    # these to build a model to impute missing values
    CT_valid = household.CTANNUAL > 0

    # Find the mean reported Council Tax bill for a given
    # (region, CT band, is-single-person-household) triplet
    region = household.GVTREGNO[CT_valid]
    band = household.CTBAND[CT_valid]
    single_person = (household.ADULTH == 1)[CT_valid]
    ctannual = household.CTANNUAL[CT_valid]

    # Build the table
    CT_mean = ctannual.groupby(
        [region, band, single_person], dropna=False
    ).mean()
    CT_mean = CT_mean.replace(-1, CT_mean.mean())

    # For every household consult the table to find the imputed
    # Council Tax bill
    pairs = household.set_index(
        [household.GVTREGNO, household.CTBAND, (household.ADULTH == 1)]
    )
    hh_CT_mean = pd.Series(index=pairs.index)
    has_mean = pairs.index.isin(CT_mean.index)
    hh_CT_mean[has_mean] = CT_mean[pairs.index[has_mean]].values
    hh_CT_mean[~has_mean] = 0
    CT_imputed = hh_CT_mean

    # For households which originally reported Council Tax,
    # use the reported value. Otherwise, use the imputed value
    council_tax = pd.Series(
        np.where(
            # 2018 FRS uses blanks for missing values, 2019 FRS
            # uses -1 for missing values
            (household.CTANNUAL < 0) | household.CTANNUAL.isna(),
            max_(CT_imputed, 0).values,
            household.CTANNUAL,
        )
    ).values
    tables["household"]["council_tax"] = council_tax
    BANDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # Band 1 is the most common
    tables["household"]["council_tax_band"] = categorical(
        household.CTBAND, 1, range(1, 10), BANDS
    )
    # Domestic rates variables are all weeklyised, unlike Council Tax variables (despite the variable name suggesting otherwise)
    tables["household"]["domestic_rates"] = (
        np.select(
            [
                household.RTANNUAL >= 0,
                household.RT2REBAM >= 0,
                True,
            ],
            [
                household.RTANNUAL,
                household.RT2REBAM,
                0,
            ],
        )
        * 52
    )
    return tables


def add_market_income(
    tables: Dict[str, DataFrame],
    person: DataFrame,
    pension: DataFrame,
    account: DataFrame,
    household: DataFrame,
    oddjob: DataFrame,
):
    """Adds income variables (non-benefit).

    Args:
        tables (Dict[str, DataFrame])
        person (DataFrame)
        pension (DataFrame)
        job (DataFrame)
        account (DataFrame)
        household (DataFrame)
        oddjob (DataFrame)
    """
    tables["person"]["employment_income"] = person.INEARNS * 52

    pension_payment = sum_to_entity(
        pension.PENPAY * (pension.PENPAY > 0), pension.person_id, person.index
    )
    pension_tax_paid = sum_to_entity(
        (pension.PTAMT * ((pension.PTINC == 2) & (pension.PTAMT > 0))),
        pension.person_id,
        person.index,
    )
    pension_deductions_removed = sum_to_entity(
        pension.POAMT
        * (
            ((pension.POINC == 2) | (pension.PENOTH == 1))
            & (pension.POAMT > 0)
        ),
        pension.person_id,
        person.index,
    )

    tables["person"]["pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * 52

    tables["person"]["self_employment_income"] = person.SEINCAM2 * 52

    INVERTED_BASIC_RATE = 1.25

    tables["person"]["tax_free_savings_income"] = (
        sum_to_entity(
            account.ACCINT * (account.ACCOUNT == 21),
            account.person_id,
            person.index,
        )
        * 52
    )
    taxable_savings_interest = (
        sum_to_entity(
            (
                account.ACCINT
                * np.where(account.ACCTAX == 1, INVERTED_BASIC_RATE, 1)
            )
            * (account.ACCOUNT.isin((1, 3, 5, 27, 28))),
            account.person_id,
            person.index,
        )
        * 52
    )
    tables["person"]["savings_interest_income"] = (
        taxable_savings_interest
        + tables["person"]["tax_free_savings_income"][...]
    )
    tables["person"]["dividend_income"] = (
        sum_to_entity(
            (
                account.ACCINT
                * np.where(account.INVTAX == 1, INVERTED_BASIC_RATE, 1)
            )
            * (
                ((account.ACCOUNT == 6) & (account.INVTAX == 1))
                | account.ACCOUNT.isin((7, 8))
            ),  # GGES  # Stocks/shares/UITs
            account.person_id,
            person.index,
        )
        * 52
    )
    is_head = person.HRPID == 1
    household_property_income = (
        household.TENTYP2.isin((5, 6)) * household.SUBRENT
    )  # Owned and subletting
    persons_household_property_income = pd.Series(
        household_property_income[person.household_id].values,
        index=person.index,
    ).fillna(0)
    tables["person"]["property_income"] = (
        max_(
            0,
            is_head * persons_household_property_income
            + person.CVPAY
            + person.ROYYR1,
        )
        * 52
    )
    maintenance_to_self = max_(
        pd.Series(
            where(person.MNTUS1 == 2, person.MNTUSAM1, person.MNTAMT1)
        ).fillna(0),
        0,
    )
    use_DWP_usual_amount = person.MNTUS2 == 2
    maintenance_from_DWP = pd.Series(
        where(use_DWP_usual_amount, person.MNTUSAM2, person.MNTAMT2)
    )
    tables["person"]["maintenance_income"] = (
        sum_positive_variables([maintenance_to_self, maintenance_from_DWP])
        * 52
    )

    odd_job_income = sum_to_entity(
        oddjob.OJAMT * (oddjob.OJNOW == 1), oddjob.person_id, person.index
    )

    MISC_INCOME_FIELDS = [
        "ALLPAY2",
        "ROYYR2",
        "ROYYR3",
        "ROYYR4",
        "CHAMTERN",
        "CHAMTTST",
    ]

    tables["person"]["miscellaneous_income"] = (
        odd_job_income + sum_from_positive_fields(person, MISC_INCOME_FIELDS)
    ) * 52

    PRIVATE_TRANSFER_INCOME_FIELDS = [
        "APAMT",
        "APDAMT",
        "PAREAMT",
        "ALLPAY1",
        "ALLPAY3",
        "ALLPAY4",
    ]

    tables["person"]["private_transfer_income"] = (
        sum_from_positive_fields(person, PRIVATE_TRANSFER_INCOME_FIELDS) * 52
    )

    tables["person"]["lump_sum_income"] = person.REDAMT
    return tables


def add_benefit_income(
    tables: Dict[str, DataFrame],
    person: DataFrame,
    benefits: DataFrame,
    household: DataFrame,
):
    """Adds benefit variables.

    Args:
        tables (Dict[str, DataFrame])
        person (DataFrame)
        benefits (DataFrame)
        household (DataFrame)
    """
    BENEFIT_CODES = dict(
        child_benefit=3,
        income_support=19,
        housing_benefit=94,
        AA=12,
        DLA_SC=1,
        DLA_M=2,
        IIDB=15,
        carers_allowance=13,
        SDA=10,
        AFCS=8,
        maternity_allowance=21,
        ssmg=22,
        pension_credit=4,
        child_tax_credit=91,
        working_tax_credit=90,
        state_pension=5,
        winter_fuel_allowance=62,
        incapacity_benefit=17,
        universal_credit=95,
        PIP_M=97,
        PIP_DL=96,
    )

    for benefit, code in BENEFIT_CODES.items():
        tables["person"][benefit + "_reported"] = (
            sum_to_entity(
                benefits.BENAMT * (benefits.BENEFIT == code),
                benefits.person_id,
                person.index,
            )
            * 52
        )

    tables["person"]["JSA_contrib_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((1, 3)))
            * (benefits.BENEFIT == 14),
            benefits.person_id,
            person.index,
        )
        * 52
    )
    tables["person"]["JSA_income_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((2, 4)))
            * (benefits.BENEFIT == 14),
            benefits.person_id,
            person.index,
        )
        * 52
    )
    tables["person"]["ESA_contrib_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((1, 3)))
            * (benefits.BENEFIT == 16),
            benefits.person_id,
            person.index,
        )
        * 52
    )
    tables["person"]["ESA_income_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((2, 4)))
            * (benefits.BENEFIT == 16),
            benefits.person_id,
            person.index,
        )
        * 52
    )

    tables["person"]["BSP_reported"] = (
        sum_to_entity(
            benefits.BENAMT * (benefits.BENEFIT.isin((6, 9))),
            benefits.person_id,
            person.index,
        )
        * 52
    )

    tables["person"]["winter_fuel_allowance_reported"] = (
        np.array(tables["person"]["winter_fuel_allowance_reported"]) / 52
    )

    tables["person"]["SSP"] = person.SSPADJ * 52
    tables["person"]["SMP"] = person.SMPADJ * 52

    tables["person"]["student_loans"] = np.maximum(person.TUBORR, 0)

    tables["person"]["adult_ema"] = fill_with_mean(person, "ADEMA", "ADEMAAMT")
    tables["person"]["child_ema"] = fill_with_mean(person, "CHEMA", "CHEMAAMT")

    tables["person"]["access_fund"] = np.maximum(person.ACCSSAMT, 0) * 52

    tables["person"]["education_grants"] = np.maximum(
        person[["GRTDIR1", "GRTDIR2"]].sum(axis=1), 0
    )

    tables["person"]["council_tax_benefit_reported"] = np.maximum(
        (person.HRPID == 1)
        * pd.Series(
            household.CTREBAMT[person.household_id].values, index=person.index
        ).fillna(0)
        * 52,
        0,
    )
    return tables


def add_expenses(
    tables: Dict[str, DataFrame],
    person: DataFrame,
    job: DataFrame,
    household: DataFrame,
    maintenance: DataFrame,
    mortgage: DataFrame,
    childcare: DataFrame,
    pen_prov: DataFrame,
):
    """Adds expense variables

    Args:
        tables (Dict[str, DataFrame])
        person (DataFrame)
        household (DataFrame)
        maintenance (DataFrame)
        mortgage (DataFrame)
        childcare (DataFrame)
        pen_prov (DataFrame)
    """
    tables["person"]["maintenance_expenses"] = (
        pd.Series(
            np.where(
                maintenance.MRUS == 2, maintenance.MRUAMT, maintenance.MRAMT
            )
        )
        .groupby(maintenance.person_id)
        .sum()
        * 52
    )
    tables["person"]["maintenance_expenses"] = tables["person"][
        "maintenance_expenses"
    ].fillna(0)

    tables["household"]["housing_costs"] = (
        np.where(
            household.GVTREGNO != 13, household.GBHSCOST, household.NIHSCOST
        )
        * 52
    )
    tables["household"]["rent"] = household.HHRENT.fillna(0) * 52
    tables["household"]["mortgage_interest_repayment"] = (
        household.MORTINT.fillna(0) * 52
    )
    mortgage_capital = np.where(
        mortgage.RMORT == 1, mortgage.RMAMT, mortgage.BORRAMT
    )
    mortgage_capital_repayment = sum_to_entity(
        mortgage_capital / mortgage.MORTEND,
        mortgage.household_id,
        household.index,
    )
    tables["household"][
        "mortgage_capital_repayment"
    ] = mortgage_capital_repayment

    tables["person"]["childcare_expenses"] = (
        sum_to_entity(
            childcare.CHAMT
            * (childcare.COST == 1)
            * (childcare.REGISTRD == 1),
            childcare.person_id,
            person.index,
        )
        * 52
    )

    tables["person"]["private_pension_contributions"] = max_(
        0,
        sum_to_entity(
            pen_prov.PENAMT[pen_prov.STEMPPEN.isin((5, 6))],
            pen_prov.person_id,
            person.index,
        ).clip(0, pen_prov.PENAMT.quantile(0.95))
        * 52,
    )
    tables["person"]["occupational_pension_contributions"] = max_(
        0,
        sum_to_entity(job.DEDUC1.fillna(0), job.person_id, person.index) * 52,
    )

    tables["household"]["housing_service_charges"] = (
        pd.DataFrame(
            [
                household[f"CHRGAMT{i}"] * (household[f"CHRGAMT{i}"] > 0)
                for i in range(1, 10)
            ]
        ).sum()
        * 52
    )
    tables["household"]["water_and_sewerage_charges"] = (
        pd.Series(
            np.where(
                household.GVTREGNO == 12,
                household.CSEWAMT + household.CWATAMTD,
                household.WATSEWRT,
            )
        )
        .fillna(0)
        .values
        * 52
    )
    return tables


def add_benunit_variables(tables: Dict[str, DataFrame], benunit: DataFrame):
    tables["benunit"]["benunit_rent"] = np.maximum(
        benunit.BURENT.fillna(0) * 52, 0
    )
    return tables


class FRS_2018_21(Survey):
    name = "frs_2018_21"
    label = "Family Resources Survey (3-year pooled) 2018-21"

    def generate(self):
        tables = concatenate_surveys(
            [
                FRS_2018_19().load_all(),
                FRS_2019_20().load_all(),
                FRS_2020_21().load_all(),
            ]
        )

        tables["household"]["household_weight"] = (
            tables["household"]["household_weight"] / 3
        )
        self.save(tables)


PERSON_COLUMNS = [
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "age",
    "gender",
    "employment_income",
    "self_employment_income",
    "pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "state_pension",
    "total_NI",
    "income_tax",
    "tax_band",
    "region",
    "country",
    "adjusted_net_income",
    "employment_status",
]

BENUNIT_COLUMNS = [
    "benunit_id",
    "household_id",
    "child_benefit",
    "child_tax_credit",
    "working_tax_credit",
    "housing_benefit",
    "ESA_income",
    "housing_benefit",
    "income_support",
    "JSA_income",
    "pension_credit",
    "tax_credits",
    "universal_credit",
    "working_tax_credit",
]

HOUSEHOLD_COLUMNS = [
    "household_id",
    "household_weight",
    "region",
    "council_tax_less_benefit",
    "council_tax_band",
    "ons_tenure_type",
    "people",
    "child_benefit",
    "child_tax_credit",
    "working_tax_credit",
    "housing_benefit",
    "ESA_income",
    "housing_benefit",
    "income_support",
    "JSA_income",
    "pension_credit",
    "tax_credits",
    "universal_credit",
    "working_tax_credit",
    "employment_income",
    "self_employment_income",
    "pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "state_pension",
    "total_NI",
    "country",
    "income_tax",
]


class OutputSurvey(Survey):
    """
    A survey derived by inputting another survey into a microsimulation model and retrieving its outputs.
    """

    output_year: int
    input_survey: Type[Survey]
    input_year: int

    def generate(self):
        from policyengine_uk import Microsimulation

        sim = Microsimulation(
            dataset=self.input_survey(),
            dataset_year=self.input_year,
        )

        frs_person_df = pd.DataFrame(
            sim.calculate_dataframe(PERSON_COLUMNS, period=self.output_year)
        )
        frs_benunit_df = pd.DataFrame(
            sim.calculate_dataframe(BENUNIT_COLUMNS, period=self.output_year)
        )
        frs_household_df = pd.DataFrame(
            sim.calculate_dataframe(HOUSEHOLD_COLUMNS, period=self.output_year)
        )

        for personal_variable in PERSON_COLUMNS:
            # Add a participants column to the household dataframe, if numeric
            if "float" in str(
                frs_person_df[personal_variable].dtype
            ) or "int" in str(frs_person_df[personal_variable].dtype):
                frs_person_df[f"{personal_variable}_participants"] = (
                    frs_person_df[personal_variable] > 0
                )
                frs_household_df[
                    f"{personal_variable}_participants"
                ] = frs_person_df.groupby("person_household_id")[
                    f"{personal_variable}_participants"
                ].transform(
                    "sum"
                )

        for benunit_variable in BENUNIT_COLUMNS:
            # Add a participants column to the household dataframe, if numeric
            if "float" in str(
                frs_benunit_df[benunit_variable].dtype
            ) or "int" in str(frs_benunit_df[benunit_variable].dtype):
                frs_benunit_df[f"{benunit_variable}_participants"] = (
                    frs_benunit_df[benunit_variable] > 0
                )
                frs_household_df[
                    f"{benunit_variable}_participants"
                ] = frs_benunit_df.groupby("household_id")[
                    f"{benunit_variable}_participants"
                ].transform(
                    "sum"
                )

        for household_variable in HOUSEHOLD_COLUMNS:
            # Add a participants column to the household dataframe, if numeric
            if (
                "float" in str(frs_household_df[household_variable].dtype)
                or "int" in str(frs_household_df[household_variable].dtype)
            ) and "_participants" not in household_variable:
                frs_household_df[f"{household_variable}_participants"] = (
                    frs_household_df[household_variable] > 0
                )

        tables = dict(
            person=frs_person_df,
            benunit=frs_benunit_df,
            household=frs_household_df,
        )

        self.save(tables)


class FRS_2020_OUT_22(OutputSurvey):
    name = "frs_2020_out_22"
    label = "Family Resources Survey 2020, aged to 2022"
    input_survey = FRS_2020_21
    input_year = 2020
    output_year = 2022


class FRS_2018_21_OUT_22(OutputSurvey):
    name = "frs_2018_21_out_22"
    label = "Family Resources Survey 2018-21, aged to 2022"
    input_survey = FRS_2018_21
    input_year = 2019
    output_year = 2022
