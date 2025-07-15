import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CrashDataProcessor:
    def __init__(self, crashes_path, people_path, vehicles_path):
        self.crashes_path = crashes_path
        self.people_path = people_path
        self.vehicles_path = vehicles_path
        self.df = None

    def load_and_merge(self):
        crashes_cols = [
            "CRASH_RECORD_ID", "CRASH_DATE", "POSTED_SPEED_LIMIT", "TRAFFIC_CONTROL_DEVICE",
            "DEVICE_CONDITION", "WEATHER_CONDITION", "LIGHTING_CONDITION", "CRASH_TYPE",
            "TRAFFICWAY_TYPE", "DAMAGE", "ALIGNMENT", "ROADWAY_SURFACE_COND", "NUM_UNITS",
            "MOST_SEVERE_INJURY", "INJURIES_TOTAL", "INJURIES_FATAL", "INJURIES_INCAPACITATING",
            "INJURIES_NON_INCAPACITATING", "INJURIES_REPORTED_NOT_EVIDENT", "INJURIES_NO_INDICATION",
            "CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH"
        ]

        people_cols = [
            "PERSON_ID", "PERSON_TYPE", "CRASH_RECORD_ID", "VEHICLE_ID", "CITY", "STATE",
            "SEX", "AGE", "SAFETY_EQUIPMENT", "AIRBAG_DEPLOYED", "INJURY_CLASSIFICATION",
            "PHYSICAL_CONDITION"
        ]
        
        vehicles_cols = [
            "CRASH_UNIT_ID", "CRASH_RECORD_ID", "UNIT_NO", "UNIT_TYPE", "MAKE", "MODEL",
            "VEHICLE_YEAR", "VEHICLE_DEFECT", "VEHICLE_TYPE", "VEHICLE_USE", "OCCUPANT_CNT"
        ]

        crashes = pd.read_csv(self.crashes_path)[crashes_cols]
        people = pd.read_csv(self.people_path)[people_cols]
        vehicles = pd.read_csv(self.vehicles_path)[vehicles_cols]

        merged = pd.merge(crashes, people, on="CRASH_RECORD_ID", how="left")
        merged = pd.merge(merged, vehicles, on="CRASH_RECORD_ID", how="left")

        self.df = merged

    def inspect_missing_values(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge() first.")
        
        print(f"\nShape of DataFrame: {self.df.shape[0]:,} rows * {self.df.shape[1]} columns")

        missing_counts = self.df.isna().sum()
        missing_percents = self.df.isna().mean() * 100
        missing_percents = missing_percents[missing_percents > 0]

        print("\nMissing Value Counts (non-zero only):")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False))

        print("\nMissing Value Percentage:")
        print(missing_percents.sort_values(ascending=False).round(2))

    def visualize_missing_values_distributions(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge() first.")
        
        plt.figure(figsize=(12,6))
        sns.heatmap(self.df.isna(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show(); 

        missing_percent = self.df.isna().mean() * 100
        missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

        missing_percent.plot(kind='barh', figsize=(25, 18), color='tomato')
        plt.xlabel('Percentage of Missing Values')
        plt.title('Missing Values by Column')
        plt.show(); 

        self.df.select_dtypes(include='number').hist(bins=30, figsize=(25, 15))
        plt.suptitle('Distribution of Numeric Columns')
        plt.tight_layout()
        plt.show(); 

    def clean_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge() first.")
    
        df_clean = self.df.copy()

        # Drop rows missing key identifying or critical values
        required_cols = ["PERSON_ID", "PERSON_TYPE", "VEHICLE_ID", "CITY", "STATE", "UNIT_TYPE", "OCCUPANT_CNT"]
        df_clean.dropna(subset=required_cols, inplace=True)

        # Drop rows where AGE is missing or unrealistic
        if "AGE" in df_clean.columns:
            df_clean = df_clean[(df_clean["AGE"].notna()) & (df_clean["AGE"] >= 0) & (df_clean["AGE"] <= 95)]

        # Remove rows with "UNKNOWN" or invalid values in key categorical columns
        filters = {
            "ROADWAY_SURFACE_COND": ["UNKNOWN"],
            "CITY": ["UNKNOWN"],
            "VEHICLE_TYPE": ["UNKNOWN", "NA"],
            "TRAFFIC_CONTROL_DEVICE": ["UNKNOWN"]
        }
        for col, values in filters.items():
            if col in df_clean.columns:
                df_clean = df_clean[~df_clean[col].isin(values)]

        # Drop rows with missing values in selected categorical columns
        drop_cols = [
            "SEX", "SAFETY_EQUIPMENT", "AIRBAG_DEPLOYED", "INJURY_CLASSIFICATION",
            "PHYSICAL_CONDITION", "MAKE", "MODEL", "VEHICLE_DEFECT",
            "VEHICLE_TYPE", "VEHICLE_USE", "VEHICLE_YEAR"
        ]
        df_clean.dropna(subset=[col for col in drop_cols if col in df_clean.columns], inplace=True)

        # Drop rows where SEX == 'X'
        if "SEX" in df_clean.columns:
            df_clean = df_clean[df_clean["SEX"] != "X"]

        self.df = df_clean

        missing = df_clean.isna().sum()
        remaining = missing[missing > 0]
        if not remaining.empty:
            print("Remaining missing values:")
            print(remaining)
        else:
            print("Data cleaned. No remaining missing values.")
