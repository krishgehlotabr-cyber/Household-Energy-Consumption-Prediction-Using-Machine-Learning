import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------
# Load Model & Files
# -------------------------------
@st.cache_resource
def load_files():

    model = joblib.load("best_rf_model.pkl")
    encoder = joblib.load("encoder.pkl")
    x_test = joblib.load("x_test.pkl")
    y_test = joblib.load("y_test.pkl")

    return model, encoder, x_test, y_test


model, encoder, X_test, y_test = load_files()

# -------------------------------
# Feature Names
# -------------------------------

features = [
    "Household_ID_encoded",
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC",
    "Peak_Hours_Usage_kWh",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_7"
]

# -------------------------------
# Title
# -------------------------------

st.title("🏠 Household Energy Consumption Prediction System")

st.write("Predict household energy consumption using Random Forest Regression.")

# -------------------------------
# Sidebar - User Inputs
# -------------------------------

st.sidebar.header("Enter Household Details")

household_id = st.sidebar.text_input(
    "Household ID",
    value="HH001"
)

household_size = st.sidebar.number_input(
    "Household Size",
    min_value=1,
    max_value=20,
    value=3
)

avg_temp = st.sidebar.number_input(
    "Average Temperature (°C)",
    value=25.0
)

has_ac = st.sidebar.selectbox(
    "Has AC?",
    ["Yes", "No"]
)

peak_usage = st.sidebar.number_input(
    "Peak Hours Usage (kWh)",
    value=10.0
)

day_name = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

day_mapping = {
    "Monday":0,
    "Tuesday":1,
    "Wednesday":2,
    "Thursday":3,
    "Friday":4,
    "Saturday":5,
    "Sunday":6
}

day_of_week = day_mapping[day_name]

month = st.sidebar.selectbox(
    "Month",
    list(range(1,13))
)

is_weekend = 1 if day_of_week in [5,6] else 0

lag_1 = st.sidebar.number_input(
    "Previous Day Consumption (Lag-1)",
    value=50.0
)

lag_7 = st.sidebar.number_input(
    "Last Week Consumption (Lag-7)",
    value=55.0
)

# -------------------------------
# Encode Household ID
# -------------------------------

try:
    household_encoded = encoder.transform([household_id])[0]
except:
    household_encoded = 0

if has_ac == "Yes":
    has_ac = 1
else:
    has_ac = 0

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Energy Consumption"):

    input_data = np.array([[
        household_encoded,
        household_size,
        avg_temp,
        has_ac,
        peak_usage,
        day_of_week,
        month,
        is_weekend,
        lag_1,
        lag_7
    ]])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Energy Consumption : {prediction:.2f} kWh"
    )

    if prediction < 30:
        st.info("Low Energy Consumption")

    elif prediction < 60:
        st.warning("Normal Energy Consumption")

    else:
        st.error("High Energy Consumption")

# -------------------------------
# Model Evaluation
# -------------------------------

st.header("📊 Model Evaluation")

# Prediction on saved test data
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Show metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RMSE", f"{rmse:.3f}")

with col2:
    st.metric("R² Score", f"{r2:.3f}")

with col3:
    st.metric("MAPE", f"{mape*100:.2f}%")

# -------------------------------
# Actual vs Predicted
# -------------------------------

st.subheader("Actual vs Predicted")

fig1, ax1 = plt.subplots(figsize=(7,5))

ax1.scatter(y_test, y_pred, alpha=0.6)

ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")

st.pyplot(fig1)

# -------------------------------
# Residual Plot
# -------------------------------

residuals = y_test - y_pred

st.subheader("Residual Plot")

fig2, ax2 = plt.subplots(figsize=(7,5))

ax2.scatter(y_pred, residuals, alpha=0.6)

ax2.axhline(
    y=0,
    color="red",
    linestyle="--"
)

ax2.set_xlabel("Predicted")

ax2.set_ylabel("Residual")

st.pyplot(fig2)

# -------------------------------
# Residual Distribution
# -------------------------------

st.subheader("Residual Distribution")

fig3, ax3 = plt.subplots(figsize=(7,5))

sns.histplot(
    residuals,
    bins=30,
    kde=True,
    ax=ax3
)

ax3.axvline(
    x=0,
    color="red",
    linestyle="--"
)

st.pyplot(fig3)

# -------------------------------
# Feature Importance
# -------------------------------

st.header("Feature Importance")

importances = model.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

feat_imp = feat_imp.sort_values(
    by="Importance",
    ascending=False
)

fig4, ax4 = plt.subplots(figsize=(10,6))

ax4.barh(
    feat_imp["Feature"],
    feat_imp["Importance"]
)

ax4.set_xlabel("Importance")
ax4.set_ylabel("Feature")
ax4.set_title("Feature Importance")

st.pyplot(fig4)

st.dataframe(
    feat_imp,
    use_container_width=True
)

# -------------------------------
# Feature Importance Heatmap
# -------------------------------

st.subheader("Feature Importance Heatmap")

heatmap_df = pd.DataFrame(
    [importances],
    columns=features
)

fig5, ax5 = plt.subplots(figsize=(12,2))

sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="coolwarm",
    cbar=False,
    ax=ax5
)

st.pyplot(fig5)

# -------------------------------
# Download Results
# -------------------------------

st.header("Download Prediction Results")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": residuals
})

csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="prediction_results.csv",
    mime="text/csv"
)

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")
st.caption("Developed using Streamlit + Random Forest Regression")