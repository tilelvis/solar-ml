import pandas as pd
import numpy as np
import os

def generate_solar_data(output_path, start_date='2022-01-01', end_date='2023-12-31'):
    """
    Generates a synthetic solar panel dataset and saves it to a CSV file.

    The dataset includes timestamp, temperature, irradiance, wind_speed,
    and energy_output.

    Args:
        output_path (str): The path to save the output CSV file.
        start_date (str): The start date for the data generation.
        end_date (str): The end date for the data generation.
    """
    # Create a date range for the timestamps
    timestamps = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='H'))
    df = pd.DataFrame({'timestamp': timestamps})

    # --- Feature Generation ---

    # 1. Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear

    # 2. Temperature (°C) - Varies daily and seasonally
    seasonal_variation = -15 * np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    daily_variation = -8 * np.cos(2 * np.pi * df['hour'] / 24)
    df['temperature'] = 20 + seasonal_variation + daily_variation + np.random.normal(0, 2, len(df))

    # 3. Irradiance (W/m^2) - Solar radiation, zero at night
    # Peaks at noon, varies with season, and has random cloudiness effect
    daylight_hours = (df['hour'] > 5) & (df['hour'] < 19)
    irradiance = np.zeros(len(df))

    # Calculate peak irradiance based on season (higher in summer)
    peak_irradiance = 1000 * (1 - 0.3 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365.25)) # Peak in summer (day 172)

    # Sinusoidal pattern for daylight hours
    irradiance[daylight_hours] = peak_irradiance[daylight_hours] * np.sin(np.pi * (df['hour'][daylight_hours] - 5) / 14)

    # Add cloudiness effect (random noise)
    cloudiness = np.random.uniform(0.3, 1.0, len(df))
    df['irradiance'] = np.maximum(0, irradiance * cloudiness)

    # 4. Wind Speed (m/s)
    df['wind_speed'] = np.abs(5 + np.random.normal(0, 3, len(df)))

    # 5. Energy Output (kWh) - Target variable
    # Depends heavily on irradiance, with a small temperature dependency
    # Panel efficiency decreases slightly at very high temperatures
    efficiency = 0.18 # Average panel efficiency
    panel_area = 1.6 # m^2
    temp_factor = 1 - 0.005 * (df['temperature'] - 25) # Efficiency loss above 25°C

    # Energy output is irradiance * area * efficiency * temp_factor
    # The result is in Watts, so divide by 1000 to get kW. Since it's per hour, it's kWh.
    energy_output = (df['irradiance'] * panel_area * efficiency * temp_factor) / 1000

    # Add some random noise to the output
    noise = np.random.normal(0, 0.05, len(df))
    df['energy_output'] = np.maximum(0, energy_output + noise)

    # --- Final Touches ---

    # Drop intermediate columns
    df = df[['timestamp', 'temperature', 'irradiance', 'wind_speed', 'energy_output']]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")

if __name__ == '__main__':
    # Define the output path relative to the project structure
    # This assumes the script is run from the root of the 'solar-energy-forecasting' directory
    output_file = 'solar-energy-forecasting/data/solar_panel_data.csv'
    generate_solar_data(output_file)
