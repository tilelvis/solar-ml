# # Notebook: 02 - Model Training and Evaluation
#
# This notebook focuses on training multiple regression models and evaluating their performance
# to select the best one for forecasting solar energy output.
#
# The steps are:
# 1.  **Load the processed data**.
# 2.  **Split the data** into training and testing sets using a time-based split.
# 3.  **Define and train models**: Linear Regression, Random Forest, and XGBoost.
# 4.  **Evaluate models** using MAE, RMSE, and R².
# 5.  **Select and save the best model**.

# ## 1. Setup and Load Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Set plot style
sns.set_style("whitegrid")

# Define file paths
PROCESSED_DATA_PATH = 'solar-energy-forecasting/data/solar_panel_data_processed.csv'
MODEL_DIR = 'solar-energy-forecasting/models/'

# Load the processed dataset
df = pd.read_csv(PROCESSED_DATA_PATH, index_col='timestamp', parse_dates=True)

print("Loaded processed data:")
print(df.head())

# ## 2. Data Splitting

# For time-series data, we should split based on time, not randomly.
# We'll use the last 3 months of data for testing.
split_date = df.index.max() - pd.DateOffset(months=3)

train_df = df[df.index <= split_date]
test_df = df[df.index > split_date]

# Define features (X) and target (y)
TARGET = 'energy_output'
FEATURES = [col for col in df.columns if col != TARGET]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ## 3. Model Training

# We will train three different models to compare their performance.

# ### 3.1. Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# ### 3.2. Train and Evaluate Models
results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "model": model  # Store the trained model object
    }

    print(f"Evaluation for {name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")

# ## 4. Model Comparison

# Let's create a DataFrame to easily compare the results.
results_df = pd.DataFrame({
    "MAE": {name: res["MAE"] for name, res in results.items()},
    "RMSE": {name: res["RMSE"] for name, res in results.items()},
    "R2": {name: res["R2"] for name, res in results.items()}
})

print("\n--- Model Comparison ---")
print(results_df)

# ## 5. Select and Save the Best Model

# Based on the R² score (higher is better) and RMSE (lower is better),
# we will select the best model.
best_model_name = results_df['R2'].idxmax()
best_model = results[best_model_name]['model']

print(f"\nBest model selected: {best_model_name}")

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the best model to a file
model_path = os.path.join(MODEL_DIR, 'best_forecasting_model.joblib')
joblib.dump(best_model, model_path)

print(f"Best model saved to {model_path}")

# --- Optional: Plot Feature Importances for Tree-based Models ---
if hasattr(best_model, 'feature_importances_'):
    print("\nPlotting feature importances...")

    importances = best_model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f'Top 15 Feature Importances for {best_model_name}')
    plt.tight_layout()
    plt.show()

print("\nModel training and evaluation complete.")
