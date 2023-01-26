from survey_enhance.survey import Survey
from typing import Type
from .raw_lcfs import RawLCFS, RawLCFS_2020_21
import pandas as pd
from typing import Dict

LCFS_TO_VARIABLE = {"P601": "food_and_non_alcoholic_beverages_consumption", "P602": "alcohol_and_tobacco_consumption", "P603": "clothing_and_footwear_consumption", "P604": "housing_water_and_electricity_consumption", "P605": "household_furnishings_consumption", "P606": "health_consumption", "P607": "transport_consumption", "P608": "communication_consumption", "P609": "recreation_consumption", "P610": "education_consumption", "P611": "restaurants_and_hotels_consumption", "P612": "miscellaneous_consumption", "C72211": "petrol_spending", "C72212": "diesel_spending", "P537": "domestic_energy_consumption"}


HOUSEHOLD_LCF_RENAMES = {
    "G018": "is_adult",
    "G019": "is_child",
    "Gorx": "region",
}
PERSON_LCF_RENAMES = {
    "B303p": "employment_income",
    "B3262p": "self_employment_income",
    "B3381": "state_pension",
    "P049p": "pension_income",
}
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

COICOP_VARIABLE_NAMES = {
    1: "food_and_non_alcoholic_beverages_consumption",
    2: "alcohol_and_tobacco_consumption",
    3: "clothing_and_footwear_consumption",
    4: "housing_water_and_electricity_consumption",
    5: "household_furnishings_consumption",
    6: "health_consumption",
    7: "transport_consumption",
    8: "communication_consumption",
    9: "recreation_consumption",
    10: "education_consumption",
    11: "restaurants_and_hotels_consumption",
    12: "miscellaneous_consumption",
}

class LCFS_2020_21(Survey):
    name = "lcfs_2020_21"
    label = "Living Costs and Food Survey"

    def generate(self):
        raw_lcfs = RawLCFS_2020_21()
        household = raw_lcfs.lcfs_2020_dvhh_ukanon
        person = raw_lcfs.lcfs_2020_dvper_ukanon202021

        tables = dict(
            person=pd.DataFrame(),
            benunit=pd.DataFrame(),
            household=pd.DataFrame(),
        )

        tables = self.add_id_variables(tables, person)
        tables = self.add_consumption_variables(tables, household)
        tables = self.add_personal_variables(tables, person)
        tables = self.add_household_variables(tables, household)

        self.save(tables)

    def add_id_variables(self, tables: Dict[str, pd.DataFrame], person: pd.DataFrame):
        tables["household"]["household_id"] = person.case.unique()
        tables["person"]["person_household_id"] = person.case.values
        tables["person"]["person_id"] = tables["person"]["person_household_id"] * 10 + person.Person.values
        tables["benunit"]["benunit_id"] = person.BUMEMBER.unique()
        tables["benunit"]["person_benunit_id"] = tables["person"]["person_household_id"] * 10 + person.BUMEMBER.values

        return tables

    def add_personal_variables(self, tables: Dict[str, pd.DataFrame], person: pd.DataFrame):
        tables["person"]["age"] = person.a005p.values
        tables["person"]["gender"] = person.A004.values

        tables["person"]["employment_income"] = person.B303p.values
        tables["person"]["self_employment_income"] = person.B3262p.values
        tables["person"]["state_pension"] = person.B3381.values
        tables["person"]["pension_income"] = person.P049p.values

        return tables
    
    def add_household_variables(self, tables: Dict[str, pd.DataFrame], household: pd.DataFrame):
        tables["household"]["region"] = household.Gorx.map(REGIONS).values

        return tables

    def add_consumption_variables(self, tables: Dict[str, pd.DataFrame], household: pd.DataFrame):
        for variable, name in LCFS_TO_VARIABLE.items():
            tables["household"][name] = household[variable].values
        
        return tables
