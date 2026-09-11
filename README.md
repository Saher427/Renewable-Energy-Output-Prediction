# RenewCast — Forecasting Renewable Energy Output for Smart Grid Balancing

## Problem Statement
Can we predict how much renewable energy (solar/wind) will be available at a given time, based on weather conditions and time-based patterns, to help power grids balance supply between renewable and traditional energy sources?

## Project Overview
Renewable energy sources like solar and wind are inherently variable, as output depends heavily on time of day, season, and weather conditions. This makes it difficult for power grids to plan when to rely on renewables versus traditional backup sources. This project uses machine learning to predict renewable energy output, combining time-based patterns with real weather data, and compares multiple models to identify the best-performing approach.

## Objectives
- Predict renewable energy production (in MWh) using time-based and weather-based features
- Compare model performance across different algorithms (baseline vs. boosting-based advanced models)
- Identify which factors most strongly influence solar vs. wind output

## Dataset
**Wind & Solar Energy Production Dataset (with Weather Features)**
- Source: Kaggle
- 51,862 hourly measurements (after cleaning), spanning 2020-01-01 to 2025-11-30
- Original features: `Date`, `Start_Hour`, `End_Hour`, `Source` (Wind/Solar), `Day_of_Year`, `Day_Name`, `Month_Name`, `Season`, `Production` (MWh)
- Weather features added: `Temperature_C`, `Humidity_Percent`, `Precipitation_mm`, `WindSpeed_kmh`, `Rainfall_Flag`
- Raw dataset (no weather): `Data/Raw/Energy Production Dataset.csv`
- Weather-enriched raw dataset: `Data/Modified Dataset/Dataset_with_Weather_features.csv`

Weather features were merged in to strengthen prediction accuracy and to support the project's second phase — a household renewable energy recommendation system, which will also rely on location-based weather data.

## Approach
1. **Data Preparation**: cleaning, validation, and encoding (including new weather features)
2. **Exploratory Data Analysis**: understanding patterns in production across season, hour, source, time of year, and weather conditions
3. **Baseline Modeling**: Linear Regression
4. **Advanced Modeling**: XGBoost / LightGBM (boosting), compared against the baseline
5. **Interpretability**: identifying which factors most influence output (planned)

---

## Data Preparation Summary
- Started with 51,864 rows across 9 original columns, now extended with 5 weather columns (14 total)
- Zero missing values, zero duplicate rows
- Removed 2 rows with an invalid "Mixed" `Source` category (not enough data to model meaningfully)
- Validated all numeric ranges (hours 0–23, `Day_of_Year` 1–366, non-negative `Production`)
- Validated new weather columns: `Humidity_Percent` (0–100), `Precipitation_mm` and `WindSpeed_kmh` (non-negative), `Rainfall_Flag` (Yes/No only)
- Confirmed clean, consistent categorical values across `Source`, `Season`, `Day_Name`, `Month_Name`, and `Rainfall_Flag`
- One-hot encoded categorical features and converted boolean columns to integer (0/1) format for model compatibility
- Final dataset: 51,862 clean rows, saved in two versions — a readable version for EDA and an encoded version for modeling

## EDA Summary and Key Findings
| Finding | Insight |
|---|---|
| Source Impact | Wind produces 53% more energy than Solar |
| Seasonal Pattern | Winter: 6,454 MWh (highest), Summer: 5,915 MWh (lowest) |
| Daily Pattern | Peak production at 12:00–14:00 (midday) |
| Wind Peak | 14:00 (8,303 MWh) |
| Solar Peak | 13:00 (8,476 MWh) |
| Best Predictor | Start_Hour (correlation: 0.51) |
| Day of Week | No significant difference |

**Visualizations generated** (saved in `Visualisations/EDA/`), including production distribution, categorical breakdowns, source/season/hour/day patterns, outlier checks, correlation heatmap (updated to include weather features), source × season interaction, monthly production, and year-over-year trend (2020–2025). Weather-specific charts (Production vs. Temperature, Production vs. Wind Speed, Rainy vs. Non-Rainy day comparison) are being added as part of the current update.

## Baseline Modeling Summary — Linear Regression
- Built features from the encoded dataset (`Date` dropped after extracting `Year`; `Day_of_Year`, `Month_Name`, `Season`, and weather features retained)
- Checked for multicollinearity (correlation > 0.9 threshold) — no strongly redundant feature pairs found; `Start_Hour`/`End_Hour` showed moderate correlation (0.76) but were both retained since they fell below the threshold
- 80/20 train/test split (`random_state=42`)
- Trained a Linear Regression model as the baseline

**Performance — before vs. after adding weather features:**
| Metric | Before (no weather) | After (with weather) |
|---|---|---|
| RMSE | 3804.64 MWh | 3631.05 MWh |
| MAE | 3020.00 MWh | 2883.90 MWh |
| R² | 0.0809 | 0.1629 |

**Interpretation:** Adding weather features (Temperature, Humidity, Precipitation, Wind Speed, Rainfall Flag) alongside Year nearly doubled the model's R². The actual-vs-predicted plot still shows predictions clustering around the mean rather than tracking the true range — a classic underfitting pattern — confirming that Production depends on non-linear, interactive patterns a straight-line model can't fully capture. This establishes an improved, honest benchmark for XGBoost/LightGBM to beat.

Feature coefficients show seasonal/monthly terms still dominating (`Season_Winter`, `Month_Name_March`, `Month_Name_September`, `Season_Summer`), consistent with the EDA's seasonal findings, along with `Source_Wind` and `Rainfall_Flag_Yes` appearing among the top drivers. Note: continuous weather features (Temperature, Humidity, WindSpeed) did not appear in the top coefficients by raw magnitude — this is expected, since features weren't scaled before training, which biases raw coefficient comparisons toward 0/1 encoded categorical features. SHAP-based interpretability gave a fairer, scale-independent view (see below).

Chart saved in `Visualisations/Models/`: actual-vs-predicted and residual plots.

## Advanced Modeling & Final Comparison

Random Forest and XGBoost were trained with default parameters, then XGBoost was tuned using GridSearchCV (Random Forest tuning was tested separately and found to produce results essentially identical to the untuned version, so the default Random Forest results are used in the final comparison).

**Final Model Comparison:**
| Model | RMSE (MWh) | MAE (MWh) | R² |
|---|---|---|---|
| Linear Regression | 3631.05 | 2883.90 | 0.1629 |
| Random Forest | 1991.66 | 1382.90 | 0.7481 |
| **XGBoost (Tuned)** | **1537.77** | **1126.88** | **0.8499** |

**Best Model: XGBoost (Tuned)** — achieved the lowest error and highest R² of all models tested, and is selected as the final production model for the dashboard.

## Interpretability — SHAP

SHAP (TreeExplainer) was applied to the final tuned XGBoost model. The top features by mean |SHAP value| are `Year`, `Day_of_Year`, `Start_Hour`, and `End_Hour` — the temporal features dominate, more so than the linear baseline's coefficients suggested. `Source_Wind` ranks 5th, confirming Wind's higher output vs. Solar, and weather features (`Temperature_C`, `Humidity_Percent`, `WindSpeed_kmh`) follow with moderate, fairly symmetric influence.

`Year` and `Day_of_Year` outranking the one-hot seasonal/monthly dummies makes sense for a tree-based model: `Day_of_Year` is continuous with 366 possible values, letting XGBoost find much finer seasonal splits than a blunt 4-category `Season` dummy can offer — consistent with the clear cyclical (red/blue banding) pattern seen in the SHAP plot, and with the seasonal cycle already identified in EDA.

Chart saved in `Visualisations/Models/shap_summary.png`.

## Known Limitation Being Investigated: Wind/Solar Subgroup Performance

Since Wind makes up 82% of the dataset and Solar only 18%, we checked whether the final model performs comparably on both subsets, or whether it implicitly favors Wind due to the class imbalance (a concern our EDA also flagged: *"consider separate models for Wind and Solar"*). This was evaluated by splitting test-set predictions by `Source_Wind` and computing RMSE/MAE/R² separately for each.

**Result: no bias found.** Wind (n=8,486) and Solar (n=1,887) both achieved an **identical R² of 0.8496** on the held-out test set. RMSE/MAE are lower in absolute terms for Solar, but this reflects Solar's naturally smaller production scale (EDA: Solar averages ~4,295 MWh vs. Wind's ~6,558 MWh), not weaker relative performance. No separate models or sample weighting were needed.

## Dashboard

An interactive Streamlit dashboard (`Dashboard/app.py`) was built on top of the final model, with three tabs:

- **Prediction** — enter a date, hour, and energy source, then provide weather conditions either manually (sliders) or via live weather fetched by city name (`Dashboard/weather_fetch.py`, using WeatherAPI.com's forecast endpoint). Returns predicted output in MWh, with contextual notes (e.g. peak solar hours, high wind speed).
- **Data Insights** — key EDA charts (production by source, by hour, by season, vs. wind speed).
- **Model Performance** — final model comparison table, SHAP summary plot, and the Wind/Solar fairness check finding above.

**Live weather forecasting is limited to today + 3 days ahead**, matching WeatherAPI.com's free-tier forecast window. This is also a reasonable boundary independent of the API: the model was trained on 2020–2025 data, and tree-based models like XGBoost extrapolate poorly beyond the range of values seen in training, so predictions far into the future (or with a `Year` value outside 2020–2026) should be treated with reduced confidence.

The core prediction logic (`build_feature_row()` and `predict_energy()`, in `Dashboard/Prediction_Engine.py`) mirrors exactly what's used in the training pipeline, converting simple user inputs into the model's full 30-feature format while correctly handling the one-hot encoding's dropped reference categories (Source=Solar, Season=Fall, Day=Friday, Month=April, Rainfall=No).

## Status
- Data Preparation: complete (updated with weather features)
- EDA: complete
- Baseline Model (Linear Regression): complete
- Advanced Modeling (Random Forest / XGBoost): complete
- Hyperparameter Tuning: complete — XGBoost tuned via GridSearchCV; Random Forest tuning tested, no meaningful improvement found
- Model Comparison: complete — XGBoost (Tuned) selected as final model
- SHAP Interpretability: complete
- Wind/Solar subgroup performance check: complete — no bias found (identical R² of 0.8496 on both)
- Dashboard: complete — Prediction, Data Insights, and Model Performance tabs, with live weather API integration

---

## How to Run This Project

**1. Clone the repository**
```bash
git clone https://github.com/your-username/Renewable-Energy-Output-Prediction.git
cd Renewable-Energy-Output-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**
Place the original dataset file in:
```
Data/Raw/Energy Production Dataset.csv
```
Place the weather-merged dataset in:
```
Data/Modified Dataset/Dataset_with_Weather_features.csv
```

**4. Run the notebooks in order**
```
Notebooks/DataCleaning.ipynb   → cleans and prepares the data (including weather features)
Notebooks/EDA.ipynb            → exploratory data analysis
Notebooks/BaselineModel.ipynb  → Linear Regression baseline
```
Open each notebook (Jupyter, JupyterLab, VS Code, or Google Colab) and run all cells from top to bottom. Each notebook depends on the outputs of the previous one, so run them in sequence.

**5. Outputs**
- Cleaned datasets are saved to `Data/Cleaned/`
- Trained models and metrics are saved to `Data/ModelResults/`
- Charts/figures are saved to `Visualisations/EDA/` and `Visualisations/Models/`

**6. Run the dashboard**
```bash
cd Dashboard
pip install streamlit requests
python -m streamlit run app.py
```
Note: live weather fetching (`Dashboard/weather_fetch.py`) requires a WeatherAPI.com API key. A key is currently hardcoded as a fallback for development — replace with your own via the `WEATHERAPI_KEY` environment variable before wider distribution.

---

## Repository Structure
```
Renewable-Energy-Output-Prediction/
├── Data/
│   ├── Raw/
│   │   └── Energy Production Dataset.csv        # original dataset (no weather)
│   ├── Modified Dataset/
│   │   └── Dataset_with_Weather_features.csv    # raw dataset merged with weather data
│   ├── Cleaned/
│   │   ├── Cleaned_Readable_Data.csv            # cleaned, readable labels (for EDA)
│   │   └── Cleaned_Production_Data.csv          # cleaned, one-hot encoded (for modeling)
│   └── ModelResults/
│       ├── baseline_model_results.csv
│       ├── baseline_model_coefficients.csv
│       ├── baseline_linear_regression.pkl
│       ├── advanced_model_results.csv           # Random Forest + XGBoost (default)
│       ├── random_forest_model.pkl
│       ├── xgboost_default_model.pkl
│       ├── tuned_xgboost_results.csv
│       ├── tuned_xgboost_model.pkl              # final production model
│       └── final_model_comparison.csv
├── Notebooks/
│   ├── Complete_Renewable_Energy_Pipeline.ipynb  # full pipeline: cleaning → EDA → models → tuning → comparison → SHAP
│   ├── WeatherFeatureEngineering.ipynb           # documents how weather data was merged into the raw dataset
│   └── Prediction_Feature_Fixed.ipynb            # model loading + feature translator, developed/tested in isolation
├── Dashboard/
│   ├── app.py                                    # Streamlit dashboard (3 tabs)
│   ├── Prediction_Engine.py                      # prediction logic used by the dashboard (build_feature_row, predict_energy)
│   └── weather_fetch.py                          # live weather forecast integration (WeatherAPI.com)
├── Visualisations/
│   ├── EDA/
│   └── Models/
├── requirements.txt
└── README.md
```

`Data/ModelResults/` holds the metrics/output files and saved models (`.pkl`) for every model trained, so results can be compared consistently and reloaded without retraining (used directly by the dashboard).

---

## References
Abd Elmunim, N., Khlifi, M. A., Aldawsari, M. A., Algarni, F., Albalawi, A., Ismail, A., & Hassan, B. M. (2026). Enhancing wind and solar energy forecasting through time-series feature engineering and ensemble machine learning. *Scientific Reports*, 16, Article 15546. https://doi.org/10.1038/s41598-026-49373-7

## Acknowledgements
This project is being developed as a capstone deliverable for the AuratTech Data and AI Fellowship.