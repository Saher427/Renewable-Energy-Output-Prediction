import os
import joblib
import pandas as pd

# Model location. Run this notebook from the project/repository root.
# model_path = os.path.join("..", "Data", "ModelResults", "tuned_xgboost_model.pkl")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "Data", "ModelResults", "tuned_xgboost_model.pkl")
model_path = os.path.normpath(model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at: {model_path}. "
        "Make sure the project repository is available before running this notebook."
    )

# Exact feature order used by the trained model.
# The omitted reference categories are represented by all-zero dummy columns:
# Source=Solar, Season=Fall, Day=Friday, Month=April, Rainfall=No.
expected_features = [
    "Start_Hour", "End_Hour", "Day_of_Year", "Temperature_C", "Humidity_Percent",
    "Precipitation_mm", "WindSpeed_kmh", "Source_Wind", "Season_Spring", "Season_Summer",
    "Season_Winter", "Day_Name_Monday", "Day_Name_Saturday", "Day_Name_Sunday",
    "Day_Name_Thursday", "Day_Name_Tuesday", "Day_Name_Wednesday", "Month_Name_August",
    "Month_Name_December", "Month_Name_February", "Month_Name_January", "Month_Name_July",
    "Month_Name_June", "Month_Name_March", "Month_Name_May", "Month_Name_November",
    "Month_Name_October", "Month_Name_September", "Rainfall_Flag_Yes", "Year"
]

model = joblib.load(model_path)
    
# Confirm that the loaded model exposes feature names and that the order matches training.
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if model_features != expected_features:
        raise ValueError(
            "Feature mismatch! The model's training feature order does not match "
            "expected_features. Do not predict until this is corrected."
        )

print(f"Model loaded successfully: {model_path}")
print(f"Number of model features: {len(expected_features)}")
print("Feature order verified.")

def build_feature_row(
    date,
    start_hour,
    end_hour,
    source,
    temperature,
    humidity,
    precipitation,
    wind_speed,
    rainfall
):
    """
    Convert simple user-friendly inputs into the exact 30-column row
    expected by the trained XGBoost model.

    Important: the training pipeline used drop_first=True for categorical
    variables. Therefore, reference categories are NOT separate columns:
      - Source: Solar is the reference -> Source_Wind = 0
      - Season: Fall is the reference -> all Season_* = 0
      - Day: Friday is the reference -> all Day_Name_* = 0
      - Month: April is the reference -> all Month_Name_* = 0
      - Rainfall: No is the reference -> Rainfall_Flag_Yes = 0
    """

    # ---------- Validate simple user inputs ----------
    date = pd.to_datetime(date, errors="raise")

    if source not in {"Wind", "Solar"}:
        raise ValueError("source must be either 'Wind' or 'Solar'.")

    if rainfall not in {"Yes", "No"}:
        raise ValueError("rainfall must be either 'Yes' or 'No'.")

    if not (0 <= int(start_hour) <= 23):
        raise ValueError("start_hour must be between 0 and 23.")

    if not (0 <= int(end_hour) <= 23):
        raise ValueError("end_hour must be between 0 and 23.")

    start_hour = int(start_hour)
    end_hour = int(end_hour)

    expected_end_hour = 0 if start_hour == 23 else start_hour + 1
    if end_hour != expected_end_hour:
        raise ValueError(
            f"end_hour must follow the training convention: start_hour + 1, "
            f"wrapping to 0 when start_hour is 23. Expected {expected_end_hour}, got {end_hour}."
        )

    # ---------- Date-derived features ----------
    year = date.year
    day_of_year = date.dayofyear
    day_name = date.day_name()
    month_name = date.month_name()

    if month_name in ["December", "January", "February"]:
        season = "Winter"
    elif month_name in ["March", "April", "May"]:
        season = "Spring"
    elif month_name in ["June", "July", "August"]:
        season = "Summer"
    else:
        season = "Fall"

    # ---------- Start with numeric features ----------
    row = pd.DataFrame({
        "Start_Hour": [start_hour],
        "End_Hour": [end_hour],
        "Day_of_Year": [day_of_year],
        "Temperature_C": [float(temperature)],
        "Humidity_Percent": [float(humidity)],
        "Precipitation_mm": [float(precipitation)],
        "WindSpeed_kmh": [float(wind_speed)],
        "Source_Wind": [int(source == "Wind")],
        "Rainfall_Flag_Yes": [int(rainfall == "Yes")],
        "Year": [year]
    })

    # ---------- One-hot columns that actually exist in training ----------
    day_columns = [
        "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"
    ]
    month_columns = [
        "August", "December", "February", "January", "July", "June",
        "March", "May", "November", "October", "September"
    ]
    season_columns = ["Spring", "Summer", "Winter"]

    for day in day_columns:
        row[f"Day_Name_{day}"] = int(day_name == day)

    for month in month_columns:
        row[f"Month_Name_{month}"] = int(month_name == month)

    for s in season_columns:
        row[f"Season_{s}"] = int(season == s)

    # ---------- Critical step: force exact training order ----------
    row = row.reindex(columns=expected_features, fill_value=0)

    # Final safety check before the model sees the row.
    if list(row.columns) != expected_features:
        raise ValueError("Generated feature row does not match the model feature order.")

    return row


def predict_energy(
    date,
    start_hour,
    end_hour,
    source,
    temperature,
    humidity,
    precipitation,
    wind_speed,
    rainfall
):
    """Build the model row and return the predicted renewable energy output in MWh."""
    row = build_feature_row(
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

    return float(model.predict(row)[0])