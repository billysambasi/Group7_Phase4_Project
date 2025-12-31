# Road Crashes Analysis

## Group Members

1. **Kago Ruburu**
2. **Purity Wambui**
3. **Emmanuel Yegon**
4. **Judy Chepkemoi**
5. **Billy Sambasi**

## Problem Statement

Road traffic crashes remain a significant public safety issue often resulting in severe injuries or fatalities. To effectively reduce crash occurrences and improve safety outcomes, it is essential to understand the key factors that contribute to crash severity and frequency. This analysis aims to explore this large-scale traffic crash dataset to answer critical questions. By addressing these questions, the analysis will help identify high-risk conditions, behaviors and regions, enabling data-driven interventions and policy recommendations aimed at improving road safety and reducing crash-related injuries and fatalities.

## Analysis Objectives

1. Identify and analyze key risk factors contributing to crash severity.
2. Examine temporal and situational patterns in crash and injury severity.
3. Evaluate the impact of preventive measures on crash outcomes.

## Business Understanding

This project analyzes road traffic crash data to help public safety agencies and policymakers enhance road safety by addressing three objectives. These insights will support targeted interventions, efficient resource allocation and data-driven policies to reduce crash frequency and severity.

## Business Questions

1. Which factors are most associated with severe or fatal injuries?
2. When and where do serious crashes most frequently occur and how do injury patterns vary?
3. How do safety measures impact crash frequency and injury severity?

## Data Understanding

### Datasets

- **[cpd-traffic-crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data)**: Crash data shows information about each traffic crash on city streets within the City of Chicago limits and under the jurisdiction of Chicago Police Department (CPD).
- **[cpd-traffic-crashes-people](https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d/about_data)**: This data contains information about people involved in a crash and if any injuries were sustained.
- **[cpd-traffic-crashes-vehicles](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data)**: This dataset contains information about vehicles (or units as they are identified in crash reports) involved in a traffic crash.

The three datasets were utilized to gain a comprehensive understanding of crash data involving both individuals and vehicles. The datasets were merged into a single DataFrame containing records and features, forming the basis for subsequent data preprocessing and cleaning.

## Data Preprocessing

### Key Features

The analysis focuses on the following features from the cleaned DataFrame:

- **Crash Information**: `CRASH_RECORD_ID`, `CRASH_DATE`, `POSTED_SPEED_LIMIT`, `TRAFFIC_CONTROL_DEVICE`, `DEVICE_CONDITION`, `WEATHER_CONDITION`, `LIGHTING_CONDITION`, `CRASH_TYPE`, `TRAFFICWAY_TYPE`, `DAMAGE`, `ALIGNMENT`, `ROADWAY_SURFACE_COND`, `NUM_UNITS`
- **Injury Data**: `MOST_SEVERE_INJURY`, `INJURIES_TOTAL`, `INJURIES_FATAL`, `INJURIES_INCAPACITATING`, `INJURIES_NON_INCAPACITATING`, `INJURIES_REPORTED_NOT_EVIDENT`, `INJURIES_NO_INDICATION`, `INJURIES_UNKNOWN`
- **Temporal Features**: `CRASH_HOUR`, `CRASH_DAY_OF_WEEK`, `CRASH_MONTH`
- **Person Information**: `PERSON_ID`, `PERSON_TYPE`, `CITY`, `STATE`, `SEX`, `AGE`, `SAFETY_EQUIPMENT`, `AIRBAG_DEPLOYED`, `INJURY_CLASSIFICATION`, `PHYSICAL_CONDITION`
- **Vehicle Information**: `CRASH_UNIT_ID`, `UNIT_NO`, `UNIT_TYPE`, `MAKE`, `MODEL`, `VEHICLE_YEAR`, `VEHICLE_DEFECT`, `VEHICLE_TYPE`, `VEHICLE_USE`, `OCCUPANT_CNT`
- **Derived Features**: `AGE_missing`, `CRASH_TYPE_GROUP`

### Data Processing Steps

- Missing values are inspected and visualized using heatmaps and bar graphs
- Data cleaning is performed using the `CrashDataProcessor` class to handle missing values and outliers
- Rows with missing critical identifiers are removed
- Unrealistic age values are filtered out
- Unknown or invalid categorical values are excluded
- Binary injury classification is applied for modeling purposes

## Exploratory Data Analysis (EDA)

### Sample Visualizations

![Age distribution by injury severity](<vizzes/Age distribution by injury severity.png>)

![Age vs. injury severity](<vizzes/Age vs. injury severity.png>)

![Age vs. total injuries](<vizzes/Age vs. Total Injuries.png>)

![Correlation matrix of numerical features](<vizzes/Correlation matrix of numerical features.png>)

![Crash frequency by day of week and hour](<vizzes/Crash Frequency - Day of week vs Hour.png>)

![Injury severity by sex](<vizzes/Injury severity among Sexes.png>)

![Injury severity by roadway surface condition](<vizzes/Injury severity by roadway surface condition.png>)

![Injury severity by weather condition](<vizzes/Injury severity by weather condition.png>)

![Injury severity vs top safety equipment used](<vizzes/Injury Severity vs Top Safety equipment used.png>)

![Number of crashes per hour](<vizzes/Number of Crashes per Hour.png>)

![Posted speed limit vs. injury severity](<vizzes/Speed Limit by Injury Severity.png>)

### Insights

- Most crashes result in minor injuries; severe injuries are rare.
- Safety belts are highly effective in reducing injury severity.
- Severe injuries cluster at higher speed limits and in adverse weather conditions.
- Crash frequency peaks during rush hours and weekdays.
- Males and SUVs show slightly higher proportions of severe injuries.

## Modeling

### Model Pipeline

- **Binary Classification**: Injury severity is mapped to binary categories (SEVERE vs NON-SEVERE)
- **Preprocessing Pipeline**: Includes StandardScaler, OneHotEncoder, and PCA for dimensionality reduction
- **Models Implemented**: 
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - AdaBoost Classifier
  - XGBoost Classifier

### Model Training and Evaluation

- Grid search with cross-validation is used for hyperparameter tuning
- Model performance is evaluated using accuracy, F1 score, and confusion matrices
- Recall-focused scoring is implemented to address class imbalance
- Severe class imbalance is observed; minority classes are poorly predicted

## Model Interpretability

- **LIME (Local Interpretable Model-agnostic Explanations)** is implemented to explain individual predictions
- The `LimeExplainer` class provides interpretable explanations for model predictions
- LIME visualizations help understand feature contributions to predictions for the best-performing models
- Feature importance analysis reveals key factors influencing crash severity predictions

## Key Findings

- Most crashes result in minor injuries; severe injuries are rare
- Safety belts are highly effective in reducing injury severity
- Severe injuries cluster at higher speed limits and in adverse weather conditions
- Crash frequency peaks during rush hours and weekdays
- Males and SUVs show slightly higher proportions of severe injuries

## Recommendations

- **Visualization**: Use stacked bar charts and countplots for categorical comparisons; use boxplots and heatmaps for numeric and correlation analysis
- **Model Improvement**: Address class imbalance for improved minority class detection (e.g., SMOTE, class weighting)
- **Interpretability**: Use LIME for model interpretability in presentations and policy recommendations
- **Safety Interventions**: Focus on high-risk conditions (adverse weather, high speed limits) and promote safety equipment usage

## Project Structure

```
Group7_Phase4_Project/
├── data/
│   ├── cpd-traffic-crashes-crashes.csv
│   ├── cpd-traffic-crashes-people.csv
│   └── cpd-trafiic-crashes-vehicles.csv
├── vizzes/
│   └── [Various visualization files]
├── index.ipynb                 # Main analysis notebook
├── utils.py                    # Data processing and modeling utilities
├── data_report.pdf            # Data analysis report
├── presentation.pdf           # Project presentation
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

## Usage

1. **Main Analysis**: Run `index.ipynb` for complete analysis and visualization
2. **Data Processing**: Use the `CrashDataProcessor` class from `utils.py` for data loading, merging, and cleaning
3. **Modeling**: Utilize the modeling classes (`BaseModelPipeline`, `ModelWithGridSearch`, `RecallGridSearch`) for machine learning workflows
4. **Interpretability**: Use the `LimeExplainer` class for model explanation and feature importance analysis

## Dependencies

- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- lime

## Files Description

- **`index.ipynb`**: Complete analysis workflow including EDA, modeling, and evaluation
- **`utils.py`**: Contains `CrashDataProcessor` and modeling utility classes
- **`data_report.pdf`**: Detailed data analysis report
- **`presentation.pdf`**: Project presentation slides

## License

MIT License. See [LICENSE](LICENSE) for details.