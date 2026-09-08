"""
Renewable Energy Output Prediction Dashboard
Aurat Tech Data and AI Fellowship — Capstone Project

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from PIL import Image

from weather_fetch import fetch_weather_forecast

st.set_page_config(page_title="Renewable Energy Predictor", page_icon="⚡", layout="wide")

# ==================================================================
# LOAD MODEL (cached so it only loads once)
# ==================================================================
@st.cache_resource
def load_model():
    return joblib.load("../Data/ModelResults/tuned_xgboost_model.pkl")

model = load_model()

MODEL_FEATURES = [
    'Start_Hour', 'End_Hour', 'Day_of_Year', 'Temperature_C', 'Humidity_Percent',
    'Precipitation_mm', 'WindSpeed_kmh', 'Source_Wind', 'Season_Spring', 'Season_Summer',
    'Season_Winter', 'Day_Name_Monday', 'Day_Name_Saturday', 'Day_Name_Sunday',
    'Day_Name_Thursday', 'Day_Name_Tuesday', 'Day_Name_Wednesday', 'Month_Name_August',
    'Month_Name_December', 'Month_Name_February', 'Month_Name_January', 'Month_Name_July',
    'Month_Name_June', 'Month_Name_March', 'Month_Name_May', 'Month_Name_November',
    'Month_Name_October', 'Month_Name_September', 'Rainfall_Flag_Yes', 'Year'
]

# ==================================================================
# TRANSLATOR FUNCTION (Member 2's work — reused as-is)
# ==================================================================
def build_feature_row(date, start_hour, source, temperature, humidity, precipitation, windspeed, rainfall):
    end_hour = 0 if start_hour == 23 else start_hour + 1
    day_of_year = date.timetuple().tm_yday
    year = date.year
    month_name = date.strftime("%B")
    day_name = date.strftime("%A")

    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    }
    season = month_to_season[date.month]

    row = {col: 0 for col in MODEL_FEATURES}
    row['Start_Hour'] = start_hour
    row['End_Hour'] = end_hour
    row['Day_of_Year'] = day_of_year
    row['Year'] = year
    row['Temperature_C'] = temperature
    row['Humidity_Percent'] = humidity
    row['Precipitation_mm'] = precipitation
    row['WindSpeed_kmh'] = windspeed

    if source == "Wind":
        row['Source_Wind'] = 1

    season_col = f"Season_{season}"
    if season_col in row:
        row[season_col] = 1

    day_col = f"Day_Name_{day_name}"
    if day_col in row:
        row[day_col] = 1

    month_col = f"Month_Name_{month_name}"
    if month_col in row:
        row[month_col] = 1

    if rainfall == "Yes":
        row['Rainfall_Flag_Yes'] = 1

    return pd.DataFrame([row])[MODEL_FEATURES]


# ==================================================================
# APP LAYOUT — TABS
# ==================================================================
st.title("⚡ Renewable Energy Output Predictor")

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Results & Insights"])

# ------------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------------
with tab1:
    st.header("Predict Energy Output")
    st.write("Enter conditions below to predict expected renewable energy production.")

    col1, col2 = st.columns(2)
    with col1:
        input_date = st.date_input(
            "Date", value=datetime.today(),
            min_value=datetime.today() - timedelta(days=365),
            max_value=datetime.today() + timedelta(days=3)  # weather API free-tier forecast limit
        )
        start_hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
        source = st.selectbox("Energy Source", ["Wind", "Solar"])
    with col2:
        rainfall_manual = st.selectbox("Is it raining? (manual mode only)", ["No", "Yes"])

    st.divider()
    st.subheader("Weather Conditions")

    input_mode = st.radio(
        "How would you like to provide weather data?",
        ["Enter manually", "Fetch by city (live forecast)"],
        horizontal=True
    )

    temperature = humidity = windspeed = precipitation = rainfall = None

    if input_mode == "Enter manually":
        w1, w2 = st.columns(2)
        with w1:
            temperature = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
            humidity = st.slider("Humidity (%)", 0, 100, 50)
        with w2:
            windspeed = st.slider("Wind Speed (km/h)", 0.0, 60.0, 15.0)
            precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
        rainfall = rainfall_manual

    else:
        city = st.text_input("Enter city name", placeholder="e.g. Lahore")
        st.caption("⚠️ Live forecast is only available up to ~3 days ahead (free-tier API limit).")

        if city:
            with st.spinner(f"Fetching forecast for {city}..."):
                weather = fetch_weather_forecast(city, input_date)

            if weather:
                temperature = weather["temperature"]
                humidity = weather["humidity"]
                windspeed = weather["windspeed"]
                precipitation = weather["precipitation"]
                rainfall = weather["rainfall"]

                st.success(f"Forecast fetched for {city} on {input_date}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Temperature", f"{temperature}°C")
                m2.metric("Humidity", f"{humidity}%")
                m3.metric("Wind Speed", f"{windspeed} km/h")
                m4.metric("Precipitation", f"{precipitation} mm")
            else:
                st.error(
                    "Couldn't fetch forecast for that city/date. "
                    "Check the city name, or make sure the date is within the next 3 days."
                )

    st.divider()

    predict_disabled = (input_mode == "Fetch by city (live forecast)" and temperature is None)

    if st.button("Predict Energy Output", type="primary", disabled=predict_disabled):
        input_row = build_feature_row(
            input_date, start_hour, source, temperature, humidity, precipitation, windspeed, rainfall
        )
        prediction = model.predict(input_row)[0]

        st.success(f"### Predicted Production: **{prediction:,.0f} MWh**")

        with st.expander("See the exact inputs used for this prediction"):
            st.dataframe(input_row.T.rename(columns={0: "Value"}))

# ------------------------------------------------------------------
# TAB 2 — RESULTS & INSIGHTS
# ------------------------------------------------------------------
with tab2:
    st.header("Model Performance")

    try:
        comparison = pd.read_csv("../Data/ModelResults/final_model_comparison.csv")
        st.dataframe(comparison, use_container_width=True)
    except FileNotFoundError:
        st.warning("final_model_comparison.csv not found — check the file path.")

    st.subheader("R² Comparison")
    try:
        chart_img = Image.open("../Visualisations/Models/model_comparison_r2.png")
        st.image(chart_img, use_container_width=True)
    except FileNotFoundError:
        st.warning("model_comparison_r2.png not found — check the file path.")

    st.divider()

    st.header("What Drives the Predictions? (SHAP)")
    try:
        shap_img = Image.open("../Visualisations/Models/shap_summary.png")
        st.image(shap_img, use_container_width=True)
        st.caption(
            "Year and Day_of_Year rank highest, followed by Start_Hour/End_Hour and Source_Wind — "
            "confirming time-based and source patterns are the strongest drivers of predicted output."
        )
    except FileNotFoundError:
        st.warning("shap_summary.png not found — check the file path.")

    st.divider()

    st.header("Key Data Insights")
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        try:
            img = Image.open("../Visualisations/EDA/3_production_by_source.png")
            st.image(img, caption="Production by Source", use_container_width=True)
        except FileNotFoundError:
            st.warning("Chart not found: 3_production_by_source.png")
    with eda_col2:
        try:
            img = Image.open("../Visualisations/EDA/5_production_by_season.png")
            st.image(img, caption="Production by Season", use_container_width=True)
        except FileNotFoundError:
            st.warning("Chart not found: 5_production_by_season.png")

    st.subheader("Wind vs. Solar — Fairness Check")
    st.write(
        "Despite Wind representing 82% of our data, the model achieves an **identical R² of 0.85** "
        "on both Wind and Solar subsets in our held-out test set — confirming no subgroup performance bias."
    )