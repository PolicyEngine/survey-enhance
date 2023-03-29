import streamlit as st

from survey_enhance.impute import Imputation
import pandas as pd

model = Imputation.load("imputations/income.pkl")

st.title("Imputing high incomes from the SPI")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    region = st.selectbox(
        "Region",
        [
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
        ],
    )
    gender = st.selectbox(
        "Gender",
        [
            "MALE",
            "FEMALE",
        ],
    )
    mean_quantile = st.slider(
        "Mean quantile", min_value=0.0, max_value=1.0, value=0.5
    )

df = pd.concat(
    [
        model.predict([[age, gender, region]], mean_quantile=mean_quantile)
        for _ in range(200)
    ]
)

INTERVAL_SIZE = 2_000

IMPUTATIONS = [
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "pension_income",
    "employment_expenses",
    "property_income",
    "gift_aid",
    "pension_contributions",
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
            for lower_bound in range(0, 200_000, INTERVAL_SIZE)
        }
    ).T
    count_df = count_df[count_df.columns[0]].values

    x = pd.Series(index=range(0, 200_000, INTERVAL_SIZE), data=count_df)
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
