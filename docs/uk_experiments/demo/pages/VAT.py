import streamlit as st

from survey_enhance.impute import Imputation
import pandas as pd

model = Imputation.load("imputations/vat.pkl")

st.title("Imputing VAT data from the ETB")

PREDICTORS = ["is_adult", "is_child", "is_SP_age", "household_net_income"]
IMPUTATIONS = ["full_rate_vat_expenditure_rate"]

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
    num_seniors = st.slider(
        "Number of seniors", min_value=0, max_value=5, value=0
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
                    num_seniors,
                    household_net_income,
                ]
            ],
            mean_quantile=mean_quantile,
        )
        for _ in range(100)
    ]
)

import numpy as np

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
                    & (df[selected_variable] < lower_bound + 0.01)
                ].shape[0]
            ]
            for lower_bound in np.linspace(0, 1, 100)
        }
    ).T
    count_df = count_df[count_df.columns[0]].values

    x = pd.Series(index=np.linspace(0, 1, 100), data=count_df)
    import plotly.express as px

    fig = px.bar(
        x,
    )
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Count",
        title=f"Imputed {selected_variable} values",
        showlegend=False,
        xaxis_tickformat=".0%",
    )

    st.plotly_chart(fig)

st.subheader(f"Full output")

st.write(df)
