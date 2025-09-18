# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Notebook: 01 - EDA, Preprocessing, and Feature Engineering
#
# This notebook covers the initial analysis of the synthetic solar panel data.
# The main steps are:
# 1.  **Load the data** from the CSV file.
# 2.  **Exploratory Data Analysis (EDA)** to understand the data's structure, distributions, and relationships.
# 3.  **Preprocessing and Feature Engineering** to prepare the data for machine learning models.

# ## 1. Setup and Load Data

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_style("whitegrid")

# Define file path
DATA_PATH = 'solar-energy-forecasting/data/solar_panel_data.csv'

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())


# ## 2. Exploratory Data Analysis (EDA)

# ### 2.1. Initial Data Inspection

# In[2]:
# Get a concise summary of the dataframe
print("\nDataset Info:")
df.info()

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert timestamp to datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'])


# ### 2.2. Visualizations

# In[3]:
# --- Time Series Plot of Energy Output ---
print("\nPlotting Energy Output over Time...")
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['energy_output'], label='Energy Output (kWh)', color='blue')
plt.title('Solar Energy Output Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Energy Output (kWh)')
plt.legend()
plt.show()

# --- Histograms of Numerical Features ---
print("Plotting histograms of numerical features...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distributions of Numerical Features')

sns.histplot(df['temperature'], ax=axes[0, 0], kde=True, color='red')
axes[0, 0].set_title('Temperature Distribution')

sns.histplot(df['irradiance'], ax=axes[0, 1], kde=True, color='orange')
axes[0, 1].set_title('Irradiance Distribution')

sns.histplot(df['wind_speed'], ax=axes[1, 0], kde=True, color='green')
axes[1, 0].set_title('Wind Speed Distribution')

sns.histplot(df['energy_output'], ax=axes[1, 1], kde=True, color='purple')
axes[1, 1].set_title('Energy Output Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ### 2.3. Relationships between Features

# In[4]:
# --- Scatter plots to check relationships ---
print("Plotting scatter plots to see feature relationships...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Irradiance vs. Energy Output
sns.scatterplot(x='irradiance', y='energy_output', data=df, ax=axes[0], alpha=0.5)
axes[0].set_title('Energy Output vs. Irradiance')

# Temperature vs. Energy Output
sns.scatterplot(x='temperature', y='energy_output', data=df, ax=axes[1], alpha=0.5)
axes[1].set_title('Energy Output vs. Temperature')

plt.show()

# --- Correlation Matrix ---
print("Plotting correlation matrix heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()


# ## 3. Preprocessing and Feature Engineering

# In[5]:
print("\nStarting preprocessing and feature engineering...")

# Set timestamp as the index
df.set_index('timestamp', inplace=True)

# --- Create Time-Based Features ---
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
df['hour'] = df.index.hour

print("Added time-based features: month, day_of_week, hour")

# --- Create Lag Features ---
# Lag features can help models capture time-series dependencies
LAG_HOURS = [1, 3, 6, 12, 24]
for lag in LAG_HOURS:
    df[f'energy_output_lag_{lag}h'] = df['energy_output'].shift(lag)

print(f"Added lag features for hours: {LAG_HOURS}")

# --- Create Rolling Average Features ---
# Rolling averages can help smooth out short-term fluctuations
ROLLING_WINDOWS = [3, 6, 12, 24]
for window in ROLLING_WINDOWS:
    df[f'temp_roll_avg_{window}h'] = df['temperature'].rolling(window=window).mean()
    df[f'irrad_roll_avg_{window}h'] = df['irradiance'].rolling(window=window).mean()

print(f"Added rolling average features for windows: {ROLLING_WINDOWS}")

# Drop rows with NaN values created by lag/rolling features
df.dropna(inplace=True)

# Display the first few rows with the new features
print("\nFirst 5 rows with new features:")
print(df.head())

# --- Save the Processed Data ---
PROCESSED_DATA_PATH = 'solar-energy-forecasting/data/solar_panel_data_processed.csv'
df.to_csv(PROCESSED_DATA_PATH)

print(f"\nProcessed data saved to {PROCESSED_DATA_PATH}")

# This concludes the EDA and preprocessing step.
# The data is now ready for model training.
