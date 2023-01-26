import streamlit as st

from survey_enhance.impute import Imputation
import pandas as pd

model = Imputation.load("wealth.pkl")

st.title("Imputing wealth from the WAS")

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
    "household_net_income",
    "num_adults",
    "num_children",
    "pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
    "num_bedrooms",
    "council_tax",
    "is_renting",
    "region",
]

IMPUTE_VARIABLES = [
    "owned_land",
    "property_wealth",
    "corporate_wealth",
    "gross_financial_wealth",
    "net_financial_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
]

col1, col2 = st.columns(2)
with col1:
    # Streamlit controls for the predictor variables.
    household_net_income = st.slider(
        "Household net income", min_value=0, max_value=100_000, value=0
    )
    num_adults = st.slider(
        "Number of adults", min_value=1, max_value=3, value=1
    )
    num_children = st.slider(
        "Number of children", min_value=0, max_value=5, value=0
    )
    pension_income = st.slider(
        "Pension income", min_value=0, max_value=100_000, value=0
    )
    employment_income = st.slider(
        "Employment income", min_value=0, max_value=100_000, value=0
    )
    self_employment_income = st.slider(
        "Self-employment income", min_value=0, max_value=100_000, value=0
    )
    capital_income = st.slider(
        "Capital income", min_value=0, max_value=100_000, value=0
    )
    num_bedrooms = st.slider(
        "Number of bedrooms", min_value=1, max_value=10, value=1
    )
    council_tax = st.slider(
        "Council tax", min_value=0, max_value=1000, value=0
    )
    is_renting = st.checkbox("Is renting")
    region = st.selectbox("Region", REGIONS)

    mean_quantile = st.slider(
        "Mean quantile", min_value=0.0, max_value=1.0, value=0.5
    )

df = pd.concat(
    [
        model.predict(
            [
                [
                    household_net_income,
                    num_adults,
                    num_children,
                    pension_income,
                    employment_income,
                    self_employment_income,
                    capital_income,
                    num_bedrooms,
                    council_tax,
                    is_renting,
                    region,
                ]
            ],
            mean_quantile=mean_quantile,
        )
        for _ in range(50)
    ]
)

INTERVAL_SIZE = 10_000

with col2:
    selected_variable = st.selectbox(
        "Variable",
        IMPUTE_VARIABLES,
    )

    count_df = pd.DataFrame(
        {
            lower_bound: [
                df[
                    (df[selected_variable] >= lower_bound)
                    & (df[selected_variable] < lower_bound + INTERVAL_SIZE)
                ].shape[0]
            ]
            for lower_bound in range(0, 1_000_000, INTERVAL_SIZE)
        }
    ).T
    count_df = count_df[count_df.columns[0]].values

    x = pd.Series(index=range(0, 1_000_000, INTERVAL_SIZE), data=count_df)
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
