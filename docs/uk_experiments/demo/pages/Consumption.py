import streamlit as st

from survey_enhance.impute import Imputation
import pandas as pd

model = Imputation.load("imputations/consumption.pkl")

st.title("Imputing consumption from the LCFS")

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
]

PREDICTOR_VARIABLES = [
    "is_adult",
    "is_child",
    "region",
    "employment_income",
    "self_employment_income",
    "pension_income",
    "household_net_income",
]

col1, col2 = st.columns(2)
with col1:
    # Adults (1-3) slider
    # Children (0-5) slider
    # Region dropdown
    # Employment income slider (0-100k)
    # Self-employment income slider (0-100k)
    # Pension income slider (0-100k)
    # Household net income slider (0-100k)

    num_adults = st.slider(
        "Number of adults", min_value=1, max_value=3, value=1
    )
    num_children = st.slider(
        "Number of children", min_value=0, max_value=5, value=0
    )
    region = st.selectbox(
        "Region",
        REGIONS,
    )
    employment_income = st.slider(
        "Employment income", min_value=0, max_value=100_000, value=0
    )
    self_employment_income = st.slider(
        "Self-employment income", min_value=0, max_value=100_000, value=0
    )
    pension_income = st.slider(
        "Pension income", min_value=0, max_value=100_000, value=0
    )
    household_net_income = st.slider(
        "Household net income", min_value=0, max_value=100_000, value=0
    )

    mean_quantile = st.slider(
        "Mean quantile", min_value=0.0, max_value=1.0, value=0.5
    )

df = pd.concat(
    [
        model.predict(
            [
                [
                    num_adults,
                    num_children,
                    region,
                    employment_income,
                    self_employment_income,
                    pension_income,
                    household_net_income,
                ]
            ],
            mean_quantile=mean_quantile,
        )
        for _ in range(50)
    ]
)

INTERVAL_SIZE = 2_000
IMPUTATIONS = [
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
    "petrol_spending",
    "diesel_spending",
    "domestic_energy_consumption",
]

with col2:
    selected_variable = st.selectbox(
        "Variable",
        IMPUTATIONS,
    )

    count_df = pd.DataFrame(
        {
            lower_bound: [
                df[
                    (df[selected_variable] >= lower_bound)
                    & (df[selected_variable] < lower_bound + INTERVAL_SIZE)
                ].shape[0]
            ]
            for lower_bound in range(0, 50_000, INTERVAL_SIZE)
        }
    ).T
    count_df = count_df[count_df.columns[0]].values

    x = pd.Series(index=range(0, 50_000, INTERVAL_SIZE), data=count_df)
    import plotly.express as px

    fig = px.bar(
        x,
    )
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Count",
        title=f"Imputed {selected_variable} values",
        showlegend=False,
        xaxis_tickformat=",.0f",
        xaxis_tickprefix="£",
    )

    st.plotly_chart(fig)

st.subheader(f"Full output")

st.write(df)
