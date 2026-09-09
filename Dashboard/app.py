# FILE: Dashboard/app.py
# STREAMLIT DASHBOARD

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from Prediction_Engine import predict_energy
from weather_fetch import fetch_weather_forecast

# ------------------------------------------------------------------
# Robust paths — resolved relative to this script's location.
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Visualisations"))

def vis_path(*parts):
    return os.path.join(VIS_DIR, *parts)

st.set_page_config(page_title="Energy Predictor", layout="wide")

# ------------------------------------------------------------------
# DESIGN SYSTEM — dark slate theme, high contrast
# ------------------------------------------------------------------
COLOR_BG = "#0F1218"
COLOR_SURFACE = "#171B24"
COLOR_SURFACE_ALT = "#1E232E"
COLOR_BORDER = "#2C3240"
COLOR_TEXT = "#F2F4F8"
COLOR_MUTED = "#9AA3B5"
COLOR_SOLAR = "#F5A623"
COLOR_WIND = "#2DD4BF"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], p, span, div, label {{
        font-family: 'Inter', sans-serif;
        color: {COLOR_TEXT} !important;
    }}

    .stApp {{ background: {COLOR_BG}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1300px; }}

    h1, h2, h3, h4 {{
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
        color: {COLOR_TEXT} !important;
    }}

    .app-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem; font-weight: 700;
        color: {COLOR_TEXT} !important; margin-bottom: 0.15rem;
    }}
    .app-subtitle {{ color: {COLOR_MUTED} !important; font-size: 0.95rem; margin-bottom: 1.2rem; }}
    .app-rule {{
        height: 3px; border: none; margin: 0 0 1.6rem 0;
        background: linear-gradient(90deg, {COLOR_SOLAR}, {COLOR_WIND} 65%, transparent);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {COLOR_BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem;
        color: {COLOR_MUTED} !important; background: transparent; padding: 0.5rem 0.1rem;
    }}
    .stTabs [aria-selected="true"] {{ color: {COLOR_TEXT} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background: linear-gradient(90deg, {COLOR_SOLAR}, {COLOR_WIND}); height: 3px;
    }}

    /* Style Streamlit's native bordered containers as our panels — no raw div hacks */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {COLOR_SURFACE} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {{
        background-color: {COLOR_SURFACE_ALT} !important;
        border: 1px solid {COLOR_BORDER} !important;
        color: {COLOR_TEXT} !important;
        border-radius: 7px !important;
    }}

    /* Date input kept light (matches its native calendar popup, which we can't fully re-theme) */
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stDateInput"] div[data-baseweb="base-input"],
    [data-testid="stDateInput"] input {{
        background-color: #FFFFFF !important;
        color: #1A2233 !important;
    }}
    [data-testid="stDateInput"] input {{
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 7px !important;
        -webkit-text-fill-color: #1A2233 !important;
    }}
    [data-testid="stDateInput"] svg {{
        fill: #1A2233 !important;
    }}
    [data-testid="stSlider"] [role="slider"] {{ background-color: {COLOR_WIND} !important; }}
    .stSlider [data-baseweb="slider"] > div > div {{ background: {COLOR_WIND} !important; }}

    label, .stSlider label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stRadio label, .stTextInput label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {{
        color: {COLOR_MUTED} !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }}

    div.stButton > button {{
        background: linear-gradient(90deg, {COLOR_SOLAR}, {COLOR_WIND});
        color: #0A0E17; font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        border: none; border-radius: 8px; padding: 0.75rem 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px); box-shadow: 0 6px 18px rgba(45, 212, 191, 0.25); color: #0A0E17;
    }}
    div.stButton > button p {{ color: #0A0E17 !important; }}

    .hero {{
        border-radius: 12px; padding: 2.2rem 1.8rem;
        border: 2px solid var(--hero-accent, {COLOR_WIND});
        background: {COLOR_SURFACE};
        height: 100%;
        display: flex; flex-direction: column; justify-content: center;
    }}
    .hero-label {{ color: {COLOR_MUTED} !important; font-size: 0.9rem; margin-bottom: 0.4rem; font-weight: 600; }}
    .hero-value {{
        font-family: 'Space Grotesk', sans-serif; font-size: 3.2rem; font-weight: 700;
        color: var(--hero-accent, {COLOR_WIND}) !important; line-height: 1.1;
    }}
    .hero-unit {{ font-size: 1.1rem; color: {COLOR_MUTED} !important; font-weight: 500; }}

    .empty-state {{
        border: 1px dashed {COLOR_BORDER}; border-radius: 12px; padding: 2.6rem 1.8rem;
        text-align: center; color: {COLOR_MUTED} !important; height: 100%;
        display: flex; align-items: center; justify-content: center;
    }}

    div[data-testid="stAlert"] {{ border-radius: 8px; }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_TEXT} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {COLOR_MUTED} !important; }}
    hr {{ border-color: {COLOR_BORDER} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Renewable Energy Output Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Forecast wind and solar output from manual or live weather inputs</div>', unsafe_allow_html=True)
st.markdown('<hr class="app-rule">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Model Performance"])

# ==================================================================
# TAB 1: PREDICTION — full-width horizontal bands using st.container(border=True)
# ==================================================================
with tab1:
    st.header("Predict Energy Output")

    # ---- BAND 1: When & Source ----
    with st.container(border=True):
        st.markdown("#### When & Source")
        b1c1, b1c2, b1c3, b1c4 = st.columns([1, 1.3, 1, 1.4])
        with b1c1:
            date = st.date_input(
                "Date", datetime.today(),
                min_value=datetime(2020, 1, 1),
                max_value=datetime.today() + timedelta(days=3)
            )
        with b1c2:
            start_hour = st.slider("Start Hour", 0, 23, 14, key="start_hour_slider")
            end_hour = (start_hour + 1) % 24
        with b1c3:
            source = st.selectbox("Energy Source", ["Wind", "Solar"], key="source_select")
        with b1c4:
            input_mode = st.radio("Weather source", ["Enter manually", "Fetch by city (live)"], key="weather_mode")
        st.caption(
            f"Prediction window: **{start_hour:02d}:00 -> {end_hour:02d}:00**  |  "
            f"Live forecast covers today + 3 days ahead"
        )

    temperature = humidity = precipitation = wind_speed = rainfall = None

    # ---- BAND 2: Weather Conditions ----
    with st.container(border=True):
        st.markdown("#### Weather Conditions")

        if input_mode == "Enter manually":
            b2c1, b2c2, b2c3, b2c4, b2c5 = st.columns(5)
            with b2c1:
                temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5, key="manual_temp")
            with b2c2:
                humidity = st.slider("Humidity (%)", 0, 100, 60, key="manual_humidity")
            with b2c3:
                wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5, key="manual_windspeed")
            with b2c4:
                precipitation = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0, 0.1, key="manual_precip")
            with b2c5:
                rainfall = st.selectbox("Rainfall Flag", ["No", "Yes"], key="manual_rainfall")

        else:
            city_col, metrics_col = st.columns([1, 3])
            with city_col:
                city = st.text_input("City name", placeholder="e.g. Lahore", key="city_input")

            if city:
                with st.spinner(f"Fetching weather for {city}..."):
                    weather = fetch_weather_forecast(city, datetime.combine(date, datetime.min.time()))

                if weather:
                    temperature = weather["temperature"]
                    humidity = weather["humidity"]
                    wind_speed = weather["windspeed"]
                    precipitation = weather["precipitation"]
                    rainfall = weather["rainfall"]

                    with metrics_col:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Temp", f"{temperature}°C")
                        m2.metric("Humidity", f"{humidity}%")
                        m3.metric("Wind", f"{wind_speed} km/h")
                        m4.metric("Rain", rainfall)
                else:
                    st.error("Couldn't fetch weather for that city/date. Check spelling, or pick a date within the next 3 days.")
            else:
                with metrics_col:
                    st.caption("Enter a city name to fetch live weather.")

    # ---- BAND 3: Predict + Result ----
    predict_col, result_col = st.columns([1, 1.4])
    with predict_col:
        predict_btn = st.button("Predict Now", use_container_width=True, disabled=(temperature is None))

    accent = COLOR_SOLAR if source == "Solar" else COLOR_WIND

    with result_col:
        if predict_btn:
            try:
                prediction = predict_energy(
                    date=date, start_hour=start_hour, end_hour=end_hour, source=source,
                    temperature=temperature, humidity=humidity, precipitation=precipitation,
                    wind_speed=wind_speed, rainfall=rainfall
                )

                note = None
                if source == "Solar" and start_hour in [12, 13, 14]:
                    note = ("success", "Peak solar hours (12:00-14:00) — output is typically highest here.")
                elif rainfall == "Yes":
                    note = ("warning", "Rain flagged for this window — expect a drop in production.")
                elif source == "Wind" and wind_speed is not None and wind_speed > 20:
                    note = ("info", "Strong wind speed — output should be well above average.")

                st.markdown(
                    f"""
                    <div class="hero" style="--hero-accent: {accent};">
                        <div class="hero-label">Predicted energy output &middot; {source}</div>
                        <div class="hero-value">{prediction:.2f} <span class="hero-unit">MWh</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if note:
                    kind, msg = note
                    getattr(st, kind)(msg)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown(
                '<div class="empty-state">Set the weather details above, then click Predict Now.</div>',
                unsafe_allow_html=True,
            )

# ==================================================================
# TAB 2: DATA INSIGHTS
# ==================================================================
with tab2:
    st.header("Exploratory Data Analysis")
    st.caption("Key visualizations from the dataset (2020-2025)")

    col1, col2 = st.columns(2)
    with col1:
        st.image(vis_path("EDA", "3_production_by_source.png"), caption="Wind produces 53% more energy than Solar")
        st.image(vis_path("EDA", "6_production_by_hour.png"), caption="Peak production at 12:00-14:00")
    with col2:
        st.image(vis_path("EDA", "13_production_vs_windspeed.png"), caption="Production vs Wind Speed")
        st.image(vis_path("EDA", "5_production_by_season.png"), caption="Winter has highest production")

# ==================================================================
# TAB 3: MODEL PERFORMANCE
# ==================================================================
with tab3:
    st.header("Model Performance Comparison")

    st.markdown("#### Final model scores")
    st.dataframe(
        pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "XGBoost (Tuned)"],
            "RMSE": [3631.05, 1991.66, 1537.77],
            "MAE": [2883.90, 1382.90, 1126.88],
            "R2": [0.1629, 0.7481, 0.8499]
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### SHAP feature importance")
    st.image(vis_path("Models", "shap_summary.png"), caption="SHAP Summary Plot")

    st.markdown("#### Wind vs. Solar fairness check")
    st.write(
        "Despite Wind representing 82% of the training data, the model achieves an "
        "**identical R2 of 0.85** on both Wind and Solar subsets in the held-out test set — "
        "confirming no subgroup performance bias."
    )
