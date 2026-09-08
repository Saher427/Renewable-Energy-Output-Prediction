# FILE: Dashboard/app.py
# STREAMLIT DASHBOARD

import streamlit as st
import pandas as pd
from datetime import datetime

# Sirf predict_energy import karein, build_feature_row nahi
from Prediction_Engine import predict_energy

# PAGE CONFIG
st.set_page_config(page_title="Energy Predictor", layout="wide")

# DESIGN SYSTEM
COLOR_BG = "#0A0E17"
COLOR_SURFACE = "#121826"
COLOR_BORDER = "#232B3D"
COLOR_TEXT = "#E8ECF3"
COLOR_MUTED = "#8B94A7"
COLOR_SOLAR = "#F5A623"
COLOR_WIND = "#2DD4BF"
COLOR_ERROR = "#F87171"
COLOR_SUCCESS = "#4ADE80"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {COLOR_TEXT};
    }}

    .stApp {{
        background: {COLOR_BG};
    }}

    .block-container {{
        padding-top: 2.5rem;
        max-width: 1200px;
    }}

    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
    }}

    /* Masthead */
    .app-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }}
    .app-subtitle {{
        color: {COLOR_MUTED};
        font-size: 0.95rem;
        margin-bottom: 1.4rem;
    }}
    .app-rule {{
        height: 2px;
        border: none;
        margin: 0 0 1.8rem 0;
        background: linear-gradient(90deg, {COLOR_SOLAR}, {COLOR_WIND} 65%, transparent);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.6rem;
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        color: {COLOR_MUTED};
        background: transparent;
        padding: 0.4rem 0.1rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLOR_TEXT} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background: linear-gradient(90deg, {COLOR_SOLAR}, {COLOR_WIND});
        height: 2px;
    }}

    /* Panels */
    .panel {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 1.5rem 1.6rem;
    }}
    .panel h3 {{
        margin-top: 0;
        font-size: 1.05rem;
    }}

    /* Inputs */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: {COLOR_WIND} !important;
    }}
    .stSlider [data-baseweb="slider"] > div > div {{
        background: {COLOR_WIND} !important;
    }}
    div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {{
        background-color: #0E1420 !important;
        border: 1px solid {COLOR_BORDER} !important;
        color: {COLOR_TEXT} !important;
        border-radius: 7px !important;
    }}
    label, .stSlider label, .stSelectbox label, .stNumberInput label, .stDateInput label {{
        color: {COLOR_MUTED} !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}

    /* Predict button */
    div.stButton > button {{
        background: linear-gradient(90deg, {COLOR_SOLAR}, {COLOR_WIND});
        color: #0A0E17;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(45, 212, 191, 0.25);
        color: #0A0E17;
    }}

    /* Result hero card */
    .hero {{
        border-radius: 12px;
        padding: 2rem 1.8rem;
        border: 1px solid var(--hero-accent, {COLOR_WIND});
        background: linear-gradient(160deg, {COLOR_SURFACE} 0%, #0E1420 100%);
        animation: hero-in 0.35s ease;
    }}
    .hero-label {{
        color: {COLOR_MUTED};
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }}
    .hero-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: var(--hero-accent, {COLOR_WIND});
        line-height: 1.1;
    }}
    .hero-unit {{
        font-size: 1.1rem;
        color: {COLOR_MUTED};
        font-weight: 500;
    }}
    .hero-note {{
        margin-top: 1rem;
        color: {COLOR_TEXT};
        font-size: 0.92rem;
    }}
    @keyframes hero-in {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .empty-state {{
        border: 1px dashed {COLOR_BORDER};
        border-radius: 12px;
        padding: 2.4rem 1.8rem;
        text-align: center;
        color: {COLOR_MUTED};
    }}

    div[data-testid="stAlert"] {{
        border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Renewable Energy Output Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Forecast wind and solar output from live weather inputs</div>', unsafe_allow_html=True)
st.markdown('<hr class="app-rule">', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Model Performance"])

# ----------------------------------------------------------------------------
# TAB 1: PREDICTION
# ----------------------------------------------------------------------------
with tab1:
    st.header("Predict Energy Output")

    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Weather details")

        date = st.date_input("Date", datetime.today())

        start_hour = st.slider("Start Hour", 0, 23, 14)
        end_hour = (start_hour + 1) % 24
        st.caption(f"Prediction window: **{start_hour:02d}:00 → {end_hour:02d}:00** (fixed 1-hour window)")

        source = st.selectbox("Energy Source", ["Wind", "Solar"])
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        precipitation = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0, 0.1)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5)
        rainfall = st.selectbox("Rainfall Flag", ["No", "Yes"])

        predict_btn = st.button("Predict Now", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    accent = COLOR_SOLAR if source == "Solar" else COLOR_WIND

    with col_result:
        if predict_btn:
            try:
                # ✅ CORRECT: predict_energy ko raw arguments pass karein
                prediction = predict_energy(
                    date=date,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    source=source,
                    temperature=temperature,
                    humidity=humidity,
                    precipitation=precipitation,
                    wind_speed=wind_speed,
                    rainfall=rainfall
                )

                note = None
                if source == "Solar" and start_hour in [12, 13, 14]:
                    note = ("success", "Peak solar hours (12:00–14:00) — output is typically highest here.")
                elif rainfall == "Yes":
                    note = ("warning", "Rain flagged for this window — expect a drop in production.")
                elif source == "Wind" and wind_speed > 20:
                    note = ("info", "Strong wind speed — output should be well above average.")

                st.markdown(
                    f"""
                    <div class="hero" style="--hero-accent: {accent};">
                        <div class="hero-label">Predicted energy output · {source}</div>
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
                """
                <div class="empty-state">
                    Set the weather details on the left, then run a prediction<br>to see the expected output here.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# TAB 2: DATA INSIGHTS (AAP KA EDA)
# ----------------------------------------------------------------------------
with tab2:
    st.header("Exploratory Data Analysis")
    st.caption("Key visualizations from the dataset (2020–2025)")

    col1, col2 = st.columns(2)

    with col1:
        st.image("Visualisations/EDA/3_production_by_source.png", caption="Wind produces 53% more energy than Solar")
        st.image("Visualisations/EDA/6_production_by_hour.png", caption="Peak production at 12:00–14:00")

    with col2:
        st.image("Visualisations/EDA/13_production_vs_windspeed.png", caption="Production vs Wind Speed")
        st.image("Visualisations/EDA/5_production_by_season.png", caption="Winter has highest production")

# ----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# ----------------------------------------------------------------------------
with tab3:
    st.header("Model Performance Comparison")

    st.markdown("### Final model scores")
    st.dataframe(
        pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "XGBoost (Tuned)"],
            "RMSE": [3631.05, 1991.66, 1537.77],
            "MAE": [2883.90, 1382.90, 1126.88],
            "R²": [0.1629, 0.7481, 0.8499]
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### SHAP feature importance")
    st.image("Visualisations/Models/shap_summary.png", caption="SHAP Summary Plot")