"""
Wine Quality Prediction Using Machine Learning
Run: python wine_quality_analysis.py
Internet access is required to download the UCI dataset.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def main():
    df = pd.read_csv(DATA_URL, sep=";")
    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().copy()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality")
    plt.title("Distribution of Wine Quality Ratings")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    X = df.drop(columns=["quality"])
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel evaluation")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 3))
    print("R2:", round(r2_score(y_test, predictions), 3))

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    importance.plot(kind="barh", figsize=(9, 5), title="Feature Importance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
