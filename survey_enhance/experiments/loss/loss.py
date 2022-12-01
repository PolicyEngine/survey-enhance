from survey_enhance.loss import LossCategory
from survey_enhance.experiments.loss.categories.households import Households
from survey_enhance.experiments.loss.categories.populations import Populations
from survey_enhance.experiments.loss.categories.country_level_programs import country_level_programs

class Demographics(LossCategory):
    weight = 1
    subcategories = [Households, Populations]

class Programs(LossCategory):
    weight = 1
    subcategories = country_level_programs

class Loss(LossCategory):
    subcategories = [
        Demographics,
        Programs,
    ]