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


# Binary Classification Crash Modeling Classes
# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import lime
import lime.lime_tabular


# Binary Target Mapping Function

def binary_injury_classification(df):
    binary_map = {
        "FATAL": "SEVERE",
        "INCAPACITATING INJURY": "SEVERE",
        "NONINCAPACITATING INJURY": "NON-SEVERE",
        "REPORTED, NOT EVIDENT": "NON-SEVERE",
        "NO INDICATION OF INJURY": "NON-SEVERE"
    }
    df = df.copy()
    df["BINARY_INJURY"] = df["MOST_SEVERE_INJURY"].map(binary_map)
    return df


# Base Pipeline Class

class BaseModelPipeline:
    def __init__(self, model, model_name, preprocessor):
        self.model_name = model_name
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        print(f"\n{self.model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred

    def get_pipeline(self):
        return self.pipeline


# Grid Search Wrapper Class

class ModelWithGridSearch:
    def __init__(self, model_name, pipeline, param_grid):
        self.model_name = model_name
        self.grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

    def train(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        print(f"\nBest Params for {self.model_name}: {self.grid_search.best_params_}")

    def evaluate(self, X_test, y_test):
        y_pred = self.grid_search.predict(X_test)
        print(f"\n{self.model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred

    def get_pipeline(self):
        return self.grid_search.best_estimator_


# Model Result Manager

class ModelManager:
    def __init__(self):
        self.model_results = {}

    def add_model_result(self, name, y_pred, y_test):
        self.model_results[name] = {
            "pred": y_pred,
            "truth": y_test
        }

    def plot_results(self):
        scores = {
            name: accuracy_score(res["truth"], res["pred"])
            for name, res in self.model_results.items()
        }
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette='Set2')
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.grid(axis='y')
        plt.show()

    def plot_confusion_matrices(self):
        for name, res in self.model_results.items():
            cm = confusion_matrix(res["truth"], res["pred"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            plt.grid(False)
            plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, classification_report

class RecallGridSearch:
    def __init__(self, model_name, pipeline, param_grid, scoring=None):
        self.model_name = model_name
        self.grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

    def train(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        print(f"\nBest Params for {self.model_name}: {self.grid_search.best_params_}")

    def evaluate(self, X_test, y_test):
        y_pred = self.grid_search.predict(X_test)
        print(f"\n{self.model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred

    def get_pipeline(self):
        return self.grid_search.best_estimator_


# LIME Explainer Class

class LimeExplainer:
    def __init__(self, fitted_pipeline, class_names, X_train):
        self.class_names = class_names
        self.pipeline = fitted_pipeline
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.classifier = self.pipeline.named_steps['classifier']

        X_transformed = self.preprocessor.transform(X_train)
        feature_names = self.preprocessor.get_feature_names_out()

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_transformed,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

    def explain_instance(self, sample_row):
        sample_transformed = self.preprocessor.transform(sample_row)
        predict_fn = lambda x: self.classifier.predict_proba(x)
        explanation = self.explainer.explain_instance(
            data_row=sample_transformed[0],
            predict_fn=predict_fn,
            num_features=10
        )
        return explanation