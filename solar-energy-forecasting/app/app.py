import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime, time

# --- Page Configuration ---
st.set_page_config(
    page_title="Solar Energy Forecaster",
    page_icon="☀️",
    layout="wide"
)

# --- Load Assets ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model."""
    model_path = os.path.join('../models/best_forecasting_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_data():
    """Loads the raw and processed data."""
    raw_data_path = os.path.join('../data/solar_panel_data.csv')
    processed_data_path = os.path.join('../data/solar_panel_data_processed.csv')

    if not os.path.exists(raw_data_path) or not os.path.exists(processed_data_path):
        st.error("Data files not found. Please generate the data first.")
        return None, None, None

    # Load raw data for historical context and feature calculation
    raw_df = pd.read_csv(raw_data_path, parse_dates=['timestamp'])
    raw_df.set_index('timestamp', inplace=True)

    # Load processed data to get the correct feature columns
    processed_df = pd.read_csv(processed_data_path)
    feature_columns = [col for col in processed_df.columns if col not in ['timestamp', 'energy_output']]

    return raw_df, feature_columns

# Load model and data
model = load_model()
raw_df, feature_columns = load_data()


# --- Feature Engineering Function ---
def prepare_features(forecast_time, temp, irradiance, wind_speed, historical_data):
    """
    Prepares the full feature set for a single prediction point.
    """
    # Create a single-row DataFrame for the new prediction
    data = {
        'timestamp': forecast_time,
        'temperature': temp,
        'irradiance': irradiance,
        'wind_speed': wind_speed
    }
    pred_df = pd.DataFrame([data])
    pred_df.set_index('timestamp', inplace=True)

    # --- Engineer Features ---
    # 1. Time-based features
    pred_df['month'] = pred_df.index.month
    pred_df['day_of_week'] = pred_df.index.dayofweek
    pred_df['hour'] = pred_df.index.hour

    # 2. Lag and Rolling Features (requires historical data)
    # We need to get data from the last 24 hours before the forecast time
    hist_subset = historical_data[historical_data.index < forecast_time].tail(24)

    # Lag features for energy_output
    for lag in [1, 3, 6, 12, 24]:
        # We need to find the value at that lag hour from historical data
        try:
            lag_time = forecast_time - pd.DateOffset(hours=lag)
            lag_value = hist_subset.loc[lag_time]['energy_output']
        except KeyError:
            lag_value = np.nan # If not found, use NaN (will be imputed)
        pred_df[f'energy_output_lag_{lag}h'] = lag_value

    # Rolling features for temp and irradiance
    for window in [3, 6, 12, 24]:
        # Combine historical data with the new prediction point to calculate rolling mean
        combined_temp = pd.concat([hist_subset['temperature'].tail(window-1), pd.Series(temp, index=[forecast_time])])
        combined_irrad = pd.concat([hist_subset['irradiance'].tail(window-1), pd.Series(irradiance, index=[forecast_time])])

        pred_df[f'temp_roll_avg_{window}h'] = combined_temp.mean()
        pred_df[f'irrad_roll_avg_{window}h'] = combined_irrad.mean()

    # Impute any missing values (e.g., if not enough historical data for a lag)
    pred_df.fillna(method='ffill', inplace=True)
    pred_df.fillna(method='bfill', inplace=True)

    # Ensure all required columns are present and in the correct order
    pred_df = pred_df.reindex(columns=feature_columns, fill_value=0)

    return pred_df


# --- Streamlit UI ---
st.title("☀️ Solar Energy Forecasting Dashboard")
st.markdown("""
This dashboard predicts the hourly solar energy output (in kWh) based on weather forecasts.
Enter the details for the forecast below.
""")

if model is None or raw_df is None:
    st.warning("Application is not ready. Please check error messages above.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forecast Input")

        # User inputs
        d = st.date_input("Forecast Date", datetime.now())
        t = st.time_input("Forecast Time", time(12, 00))
        forecast_datetime = datetime.combine(d, t)

        temp_input = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
        irrad_input = st.slider("Irradiance (W/m²)", 0.0, 1300.0, 800.0)
        wind_input = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)

        # Prediction button
        if st.button("Get Forecast", type="primary"):
            # Prepare features
            features = prepare_features(forecast_datetime, temp_input, irrad_input, wind_input, raw_df)

            # Make prediction
            prediction = model.predict(features)[0]
            prediction = max(0, prediction) # Ensure prediction is not negative

            st.subheader("Forecast Result")
            st.metric(label="Predicted Energy Output", value=f"{prediction:.4f} kWh")

            # Display the features used for prediction
            with st.expander("View Features Used for Prediction"):
                st.dataframe(features)

    with col2:
        st.subheader("Historical Energy Output")

        # Plot historical data
        # Show data for the last week for context
        hist_to_plot = raw_df['energy_output'].tail(24 * 7)
        st.line_chart(hist_to_plot)

        # Optionally, plot the prediction on the chart
        if 'prediction' in locals():
            # Create a new DataFrame for plotting the prediction point
            pred_point = pd.DataFrame(
                {'energy_output': [prediction]},
                index=[pd.to_datetime(forecast_datetime)]
            )
            st.line_chart(pd.concat([hist_to_plot, pred_point]))
            st.caption("Chart including the new forecast point (in blue).")
