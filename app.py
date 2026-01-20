import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Vehicle Volume Forecasting", layout="centered")

st.title("🚛 Commercial Vehicle Volume Forecasting")
st.write("SARIMA-based time series forecasting")

# Load dataset
df = pd.read_csv("notebooks/ltruck_sales.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

st.subheader("📊 Historical Sales")
st.line_chart(df["sales"])

# Train SARIMA model
model = SARIMAX(
    df["sales"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
model_fit = model.fit(disp=False)

# Forecast
months = st.slider("Forecast months", 1, 36, 12)
forecast = model_fit.forecast(steps=months)

st.subheader("📈 Forecast Output")
st.line_chart(forecast)

st.success("Forecast generated successfully!")
