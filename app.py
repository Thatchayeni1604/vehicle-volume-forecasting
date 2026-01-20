import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Vehicle Volume Forecast")

model = joblib.load("model/sarima_model.pkl")

st.title("🚛 Commercial Vehicle Sales Forecasting")

months = st.slider("Months to Forecast", 1, 36, 12)

forecast = model.forecast(steps=months)

future_dates = pd.date_range(
    start=pd.Timestamp.today(),
    periods=months,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": forecast
})

st.line_chart(forecast_df.set_index("Date"))
st.dataframe(forecast_df)
st.download_button(
    label="Download Forecast as CSV",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="vehicle_sales_forecast.csv",
    mime="text/csv"
)