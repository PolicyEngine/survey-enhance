from survey_enhance.survey import Survey
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
from typing import Type, Dict


class RawSPI(Survey):
    """A `Survey` instance for the Family Resources Survey."""

    name = "raw_spi"
    label = "Survey of Personal Incomes"

    main_table_name = "put1920uk"

    def generate(self, tab_folder: Path):
        """Generate the survey data from the original TAB files.

        Args:
            tab_folder (Path): The folder containing the original TAB files.
        """

        if isinstance(tab_folder, str):
            tab_folder = Path(tab_folder)

        # Load the data
        tables = {}
        tab_file = tab_folder / f"{self.main_table_name}.tab"
        table_name = "main"
        tables[table_name] = pd.read_csv(
            tab_file,
            sep="\t",
            low_memory=False,
        ).apply(pd.to_numeric, errors="coerce")

        # Save the data
        self.save(tables)

class RawSPI_2019_20(RawSPI):
    name = "raw_spi_2019_20"
    label = "Survey of Personal Incomes 2019-20 (raw)"
    
    main_table_name = "put1920uk"


class SPI_2019_20(Survey):
    name = "spi_2019_20"
    label = "Survey of Personal Incomes"

    raw_spi: Type[RawSPI] = RawSPI_2019_20

    def generate(self):
        raw_spi = self.raw_spi()

        main = raw_spi.load("main").fillna(0)

        main = extend_spi_main_table(main)
        main["id"] = np.arange(len(main))
        tables = dict(
            person=pd.DataFrame(),
            benunit=pd.DataFrame(),
            household=pd.DataFrame(),
        )

        tables, main = add_id_variables(tables, main)
        tables, main = add_demographics(tables, main)
        tables, main = add_incomes(tables, main)

        self.save(tables)


def extend_spi_main_table(main: DataFrame) -> DataFrame:
    """Extends the main SPI table to include adults and children with
    zero income, so that the total number of people is the UK population.

    Args:
        main (DataFrame): The main SPI table.

    Returns:
        DataFrame: The modified table.
    """

    from policyengine_uk import Microsimulation
    from frs import FRS_2019_20

    sim = Microsimulation(dataset=FRS_2019_20(), dataset_year=2019)

    population_in_spi_percentage = main.FACT.sum() / sim.calc("people").sum()

    RENAMES = dict(
        pension_income="PENSION",
        self_employment_income="PROFITS",
        property_income="INCPROP",
        savings_interest_income="INCBBS",
        dividend_income="DIVIDENDS",
        blind_persons_allowance="BPADUE",
        married_couples_allowance="MCAS",
        gift_aid="GIFTAID",
        capital_allowances="CAPALL",
        deficiency_relief="DEFICIEN",
        covenanted_payments="COVNTS",
        charitable_investment_gifts="GIFTINV",
        employment_expenses="EPB",
        other_deductions="MOTHDED",
        pension_contributions="PENSRLF",
        employment_income="PAY",
        state_pension="SRP",
        miscellaneous_income="OTHERINC",
        pays_scottish_income_tax="SCOT_TXP",
        household_weight="FACT",
    )

    in_frs_and_not_spi = (
        sim.calc("total_income").rank(pct=True)
        < 1 - population_in_spi_percentage
    ).values

    missing_spi = pd.DataFrame(sim.df(list(RENAMES)))[in_frs_and_not_spi].rename(
        columns=RENAMES
    )

    LOWER = np.array([0, 16, 25, 35, 45, 55, 65, 75])
    UPPER = np.array([16, 25, 35, 45, 55, 65, 75, 80])
    CODE = np.array([1, 2, 3, 4, 5, 6, 7])
    REGIONS = {
        1: "NORTH_EAST",
        2: "NORTH_WEST",
        3: "YORKSHIRE",
        4: "EAST_MIDLANDS",
        5: "WEST_MIDLANDS",
        6: "EAST_OF_ENGLAND",
        7: "LONDON",
        8: "SOUTH_EAST",
        9: "SOUTH_WEST",
        10: "WALES",
        11: "SCOTLAND",
        12: "NORTHERN_IRELAND",
    }
    age = sim.calc("age")[in_frs_and_not_spi]
    missing_spi["AGERANGE"] = 0
    for lower, upper, code in zip(LOWER, UPPER, CODE):
        missing_spi["AGERANGE"] += np.where(
            (age < upper) & (age >= lower), code, 0
        )

    missing_spi["GORCODE"] = sim.calc("region", map_to="person")[
        in_frs_and_not_spi
    ].map({y: x for x, y in REGIONS.items()})

    return pd.concat([main, missing_spi]).fillna(0)


def add_id_variables(tables: Dict[str, DataFrame], main: DataFrame):
    tables["person"]["person_id"] = main.id
    tables["person"]["person_benunit_id"] = main.id
    tables["person"]["person_household_id"] = main.id
    tables["benunit"]["benunit_id"] = main.id
    tables["household"]["household_id"] = main.id
    return tables, main


def add_demographics(tables: Dict[str, DataFrame], main: DataFrame):
    LOWER = np.array([0, 16, 25, 35, 45, 55, 65, 75])
    UPPER = np.array([16, 25, 35, 45, 55, 65, 75, 80])
    age_range = main.AGERANGE
    tables["person"]["age"] = LOWER[age_range] + np.random.rand(len(main)) * (
        UPPER[age_range] - LOWER[age_range]
    )

    REGIONS = {
        1: "NORTH_EAST",
        2: "NORTH_WEST",
        3: "YORKSHIRE",
        4: "EAST_MIDLANDS",
        5: "WEST_MIDLANDS",
        6: "EAST_OF_ENGLAND",
        7: "LONDON",
        8: "SOUTH_EAST",
        9: "SOUTH_WEST",
        10: "WALES",
        11: "SCOTLAND",
        12: "NORTHERN_IRELAND",
    }

    tables["household"]["region"] = np.array(
        [REGIONS.get(x, "UNKNOWN") for x in main.GORCODE]
    )
    return tables, main


def add_incomes(tables: Dict[str, DataFrame], main: DataFrame):
    RENAMES = dict(
        pension_income="PENSION",
        self_employment_income="PROFITS",
        property_income="INCPROP",
        savings_interest_income="INCBBS",
        dividend_income="DIVIDENDS",
        blind_persons_allowance="BPADUE",
        married_couples_allowance="MCAS",
        gift_aid="GIFTAID",
        capital_allowances="CAPALL",
        deficiency_relief="DEFICIEN",
        covenanted_payments="COVNTS",
        charitable_investment_gifts="GIFTINV",
        employment_expenses="EPB",
        other_deductions="MOTHDED",
        pension_contributions="PENSRLF",
        state_pension="SRP",
    )
    tables["person"]["pays_scottish_income_tax"] = main.SCOT_TXP == 1
    tables["person"]["employment_income"] = main[["PAY", "EPB", "TAXTERM"]].sum(axis=1)
    tables["person"]["social_security_income"] = main[
        ["SRP", "INCPBEN", "UBISJA", "OSSBEN"]
    ].sum(axis=1)
    tables["person"]["miscellaneous_income"] = main[
        ["OTHERINV", "OTHERINC", "MOTHINC"]
    ].sum(axis=1)
    for var, key in RENAMES.items():
        tables["person"][var] = main[key]
    tables["household"]["household_weight"] = main.FACT
    return tables, main
