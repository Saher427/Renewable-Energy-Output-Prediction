# Predicting Renewable Energy Output for Smart Grid Balancing

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

Weather features were merged in to strengthen prediction accuracy and to support the project's second phase, a household renewable energy recommendation system, which will also rely on location-based weather data.

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
- Final dataset: 51,862 clean rows, saved in two versions, a readable version for EDA and an encoded version for modeling

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
| Weather Factors | WindSpeed shows positive correlation with Production; Temperature and Rainfall have weaker effects |
| Correlation Insights | Day_of_Year and Month are highly correlated (redundant, keep only one for modeling); Multicollinearity not a serious issue among key features |

**Visualizations generated** (saved in `Visualisations/EDA/`), including production distribution, categorical breakdowns, source/season/hour/day patterns, outlier checks, source × season interaction, monthly production, year-over-year trend (2020–2025), weather-specific charts (Production vs. Temperature, Production vs. Wind Speed, Rainy vs. Non-Rainy day comparison), and updated full correlation heatmap (includes all encoded categorical features) plus production correlation bar chart.

## Baseline Modeling Summary, Linear Regression
- Built features from the encoded dataset (`Date` dropped after extracting `Year`; `Day_of_Year`, `Month_Name`, `Season`, and weather features retained)
- Checked for multicollinearity (correlation > 0.9 threshold), no strongly redundant feature pairs found; `Start_Hour`/`End_Hour` showed moderate correlation (0.76) but were both retained since they fell below the threshold
- 80/20 train/test split (`random_state=42`)
- Trained a Linear Regression model as the baseline

**Performance, before vs. after adding weather features:**
| Metric | Before (no weather) | After (with weather) |
|---|---|---|
| RMSE | 3804.64 MWh | 3631.05 MWh |
| MAE | 3020.00 MWh | 2883.90 MWh |
| R² | 0.0809 | 0.1629 |

**Interpretation:** Adding weather features (Temperature, Humidity, Precipitation, Wind Speed, Rainfall Flag) alongside Year nearly doubled the model's R². The actual-vs-predicted plot still shows predictions clustering around the mean rather than tracking the true range, a classic underfitting pattern, confirming that Production depends on non-linear, interactive patterns a straight-line model can't fully capture. This establishes an improved, honest benchmark for XGBoost/LightGBM to beat.

Feature coefficients show seasonal/monthly terms still dominating (`Season_Winter`, `Month_Name_March`, `Month_Name_September`, `Season_Summer`), consistent with the EDA's seasonal findings, along with `Source_Wind` and `Rainfall_Flag_Yes` appearing among the top drivers. Note: continuous weather features (Temperature, Humidity, WindSpeed) did not appear in the top coefficients by raw magnitude, this is expected, since features weren't scaled before training, which biases raw coefficient comparisons toward 0/1 encoded categorical features. SHAP-based interpretability (planned) will give a fairer, scale-independent view of feature importance.

Chart saved in `Visualisations/Models/`: actual-vs-predicted and residual plots.

## Status
- Data Preparation: complete (updated with weather features)
- EDA: complete (weather-specific charts and updated correlation heatmap in progress)
- Baseline Model (Linear Regression): complete, updated with weather features
- Advanced Modeling (XGBoost / LightGBM): in progress

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
│       ├── baseline_model_results.csv           # RMSE / MAE / R2 per model
│       ├── baseline_model_coefficients.csv      # feature coefficients
│       └── baseline_linear_regression.pkl       # saved trained model
├── Notebooks/
│   ├── DataCleaning.ipynb
│   ├── EDA.ipynb
│   └── BaselineModel.ipynb
├── Visualisations/
│   ├── EDA/                                      # EDA plot images
│   └── Models/                                   # model diagnostic plots (actual vs predicted, residuals)
├── requirements.txt
└── README.md
```

`Data/Modified Dataset/` holds the raw dataset after weather features were merged in, but before cleaning, kept separate from `Data/Raw/` so the original source data is never overwritten.

`Data/ModelResults/` is meant to hold the metrics/output files and saved model (`.pkl`) for every model going forward (baseline and boosting), so results can be compared consistently as new models are added.

---

## References
Abd Elmunim, N., Khlifi, M. A., Aldawsari, M. A., Algarni, F., Albalawi, A., Ismail, A., & Hassan, B. M. (2026). Enhancing wind and solar energy forecasting through time-series feature engineering and ensemble machine learning. *Scientific Reports*, 16, Article 15546. https://doi.org/10.1038/s41598-026-49373-7

## Acknowledgements
This project is being developed as a capstone deliverable for the AuratTech Data and AI Fellowship.