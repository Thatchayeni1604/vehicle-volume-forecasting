import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Vehicle Volume Forecasting", layout="centered")

st.title("🚛 Commercial Vehicle Volume Forecasting")
st.write("SARIMA-based time series forecasting")

# Load data
df = pd.read_csv("notebooks/ltruck_sales.csv", index_col=0)
df.index = pd.to_datetime(df.index)

sales_col = df.columns[0]   # ✅ auto-detect column

st.subheader("📊 Historical Sales")
st.line_chart(df[sales_col])

# SARIMA model
model = SARIMAX(
    df[sales_col],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
model_fit = model.fit(disp=False)

months = st.slider("Forecast Months", 1, 36, 12)
forecast = model_fit.forecast(steps=months)

st.subheader("📈 Forecast")
st.line_chart(forecast)

st.success("Forecast generated successfully!")
