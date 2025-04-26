import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate health score
def calculate_health_score(ambient_temp, core_temp, voltage, fault_count, max_faults=10):
    """
    Calculate the health score of a grid component.
    A higher score (closer to 100) indicates better health.
    """
    # Weight factors
    temp_weight = 0.4
    voltage_weight = 0.3
    fault_weight = 0.3

    # Normalize inputs
    normalized_ambient_temp = max(0, (50 - ambient_temp) / 50)  # Assume max ambient temp = 50°C
    normalized_core_temp = max(0, (70 - core_temp) / 70)        # Assume max core temp = 70°C
    normalized_voltage = min(1, voltage / 10)                  # Assume max voltage = 10V
    normalized_faults = max(0, (max_faults - fault_count) / max_faults)

    # Compute health score
    temp_score = (normalized_ambient_temp + normalized_core_temp) / 2
    health_score = (temp_weight * temp_score +
                    voltage_weight * normalized_voltage +
                    fault_weight * normalized_faults) * 100
    return round(health_score, 2)


# Streamlit App
st.title("Smart Grid Predictive Maintenance System")
st.write("This application provides predictive maintenance, health scoring, and temperature-time analysis for grid components.")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Data Cleaning
    data.replace('#NUM!', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Convert columns to numeric
    for col in ['Ambient Temperature', 'Core Temperature', 'Voltage', 'Output']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop negative values
    data = data[data['Ambient Temperature'] >= 0]

    # Define features and target
    X = data[['Ambient Temperature', 'Core Temperature', 'Voltage']]
    y = data['Output']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model3 = LinearRegression()

    # Train RandomForestRegressor separately to get feature importances
    model1.fit(X_train, y_train)

    # Ensemble model
    ensemble_model = VotingRegressor(
        estimators=[
            ('rf', model1),
            ('gbr', model2),
            ('lr', model3)
        ]
    )

    # Train ensemble model
    ensemble_model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = ensemble_model.predict(X_test)
    st.write("### Model Performance")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Plot true vs predicted values
    st.write("### True vs Predicted Values")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="True Values", color="blue")
    plt.plot(y_pred, label="Predicted Values", color="green")
    plt.title("True vs Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Feature Importance
    st.write("### Feature Importance")
    importances = model1.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.grid()
    st.pyplot(plt)

    # Health Score Calculation
    if 'Fault Count' not in data.columns:
        data['Fault Count'] = np.random.randint(0, 10, size=len(data))  # Simulated fault count if not provided

    data['Health Score'] = data.apply(
        lambda row: calculate_health_score(
            row['Ambient Temperature'],
            row['Core Temperature'],
            row['Voltage'],
            row['Fault Count']
        ), axis=1
    )

    # Display dataset with health scores
    st.subheader("Dataset with Health Scores")
    st.dataframe(data)

    # Health Score Distribution
    st.subheader("Health Score Distribution")
    st.bar_chart(data.set_index('Output')['Health Score'])

    # Custom Prediction
    st.write("### Predict Output and Health Score for Custom Input")
    user_ambient_temp = st.slider("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    user_core_temp = st.slider("Core Temperature (°C)", min_value=0.0, max_value=70.0, value=35.0)
    user_voltage = st.slider("Voltage (V)", min_value=0.0, max_value=10.0, value=3.0)
    user_fault_count = st.slider("Fault Count", min_value=0, max_value=10, value=2)

    user_input = np.array([[user_ambient_temp, user_core_temp, user_voltage]])
    user_prediction = ensemble_model.predict(user_input)
    user_health_score = calculate_health_score(user_ambient_temp, user_core_temp, user_voltage, user_fault_count)

    st.write(f"**Predicted Output for Custom Input:** {user_prediction[0]:.2f}")
    st.write(f"**Predicted Health Score for Custom Input:** {user_health_score}")

    # Temperature-Time Graph with User Prediction
    st.write("### Temperature-Time Graph with Predicted Core Temperature")
    time = np.linspace(0, 10, 100)
    temperature = np.piecewise(
        time,
        [time < 3, (time >= 3) & (time < 7), time >= 7],
        [lambda t: 20 + 2 * t, lambda t: 40 + 4 * (t - 3), lambda t: 60 + 5 * (t - 7)]
    )

    normal_max, warning_max, critical_min = 35, 45, 50

    plt.figure(figsize=(10, 6))
    plt.plot(time, temperature, label="Component Temperature", color='b')
    plt.axhline(y=normal_max, color='green', linestyle='--', label="Normal Range Max (35°C)")
    plt.axhline(y=warning_max, color='orange', linestyle='--', label="Warning Level Max (45°C)")
    plt.axhline(y=critical_min, color='red', linestyle='--', label="Critical Level Min (50°C)")
    plt.fill_between(time, 0, normal_max, color='green', alpha=0.2, label="Normal Heating")
    plt.fill_between(time, normal_max, warning_max, color='orange', alpha=0.3, label="Warning (Hot Spot Forming)")
    plt.fill_between(time, warning_max, temperature.max(), color='red', alpha=0.4, label="Critical Point (Component Break)")
    plt.scatter([5], [user_core_temp], color="purple", label="Predicted Core Temperature", s=100, zorder=5)
    plt.xlabel("Time / Sensor Reading Points")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Rise and Hot Spot Formation in Component")
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Please upload a CSV file to proceed.")
