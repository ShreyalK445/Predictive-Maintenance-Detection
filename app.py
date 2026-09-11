import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)


# -----------------------------
# File paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "cleaned_predictive_maintenance.csv"
MODEL_FILE = BASE_DIR / "final_model.pkl"


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error("cleaned_predictive_maintenance.csv not found.")
        st.stop()

    return pd.read_csv(DATA_FILE)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        st.error("final_model.pkl not found.")
        st.stop()

    package = joblib.load(MODEL_FILE)

    if isinstance(package, dict):
        model = package["model"]
        model_name = package.get(
            "model_name",
            "Predictive Maintenance Model"
        )
        features = package.get("features", [])
        threshold = package.get("threshold", 0.5)
    else:
        model = package
        model_name = "Predictive Maintenance Model"
        features = []
        threshold = 0.5

    return model, model_name, features, threshold


df = load_data()
model, model_name, model_features, threshold = load_model()


# -----------------------------
# Title
# -----------------------------
st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown(
    "Analyze machine conditions and estimate the probability of machine failure."
)

st.divider()


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

type_options = ["All"] + sorted(
    df["Type"].dropna().unique().tolist()
)

selected_type = st.sidebar.selectbox(
    "Machine Type",
    type_options
)

status_options = ["All", "Normal", "Failure"]

selected_status = st.sidebar.selectbox(
    "Machine Status",
    status_options
)


# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df.copy()

if selected_type != "All":
    filtered_df = filtered_df[
        filtered_df["Type"] == selected_type
    ]

if selected_status == "Normal":
    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 0
    ]

elif selected_status == "Failure":
    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 1
    ]


# -----------------------------
# KPI section
# -----------------------------
total_machines = len(filtered_df)

failure_count = int(
    filtered_df["Machine failure"].sum()
)

normal_count = total_machines - failure_count

failure_rate = (
    failure_count / total_machines * 100
    if total_machines > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Machines",
    f"{total_machines:,}"
)

col2.metric(
    "Failures",
    f"{failure_count:,}"
)

col3.metric(
    "Normal",
    f"{normal_count:,}"
)

col4.metric(
    "Failure Rate",
    f"{failure_rate:.2f}%"
)


st.divider()


# -----------------------------
# Charts
# -----------------------------
if len(filtered_df) > 0:

    col1, col2 = st.columns(2)

    # Failure distribution
    with col1:

        failure_data = pd.DataFrame({
            "Status": ["Normal", "Failure"],
            "Count": [normal_count, failure_count]
        })

        fig = px.pie(
            failure_data,
            names="Status",
            values="Count",
            hole=0.45,
            title="Machine Failure Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Failure rate by machine type
    with col2:

        type_failure = (
            filtered_df
            .groupby("Type")["Machine failure"]
            .mean()
            .reset_index()
        )

        type_failure["Failure Rate (%)"] = (
            type_failure["Machine failure"] * 100
        )

        fig = px.bar(
            type_failure,
            x="Type",
            y="Failure Rate (%)",
            title="Failure Rate by Machine Type",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------
    # Sensor relationships
    # -----------------------------
    st.subheader("Machine Condition Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered_df,
            x="Tool wear [min]",
            y="Torque [Nm]",
            color="Machine failure",
            hover_data=[
                "Type",
                "Rotational speed [rpm]"
            ],
            title="Tool Wear vs Torque"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            filtered_df,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="Machine failure",
            hover_data=[
                "Type",
                "Tool wear [min]"
            ],
            title="Rotational Speed vs Torque"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Temperature relationship
    fig = px.scatter(
        filtered_df,
        x="Air temperature [K]",
        y="Process temperature [K]",
        color="Machine failure",
        hover_data=[
            "Type",
            "Tool wear [min]"
        ],
        title="Air Temperature vs Process Temperature"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


else:

    st.warning("No machines match the selected filters.")


# -----------------------------
# Model information
# -----------------------------
st.divider()

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    model_name
)

col2.metric(
    "Decision Threshold",
    f"{threshold:.2f}"
)

col3.metric(
    "Input Features",
    len(model_features)
)


# -----------------------------
# Prediction Studio
# -----------------------------
st.divider()

st.subheader("🔧 Prediction Studio")

st.write(
    "Enter the current machine readings to estimate the probability of failure."
)

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=310.0
    )

    rpm = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0.0,
        max_value=5000.0,
        value=1500.0
    )


with col2:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0.0,
        max_value=300.0,
        value=100.0
    )


# -----------------------------
# Feature engineering
# -----------------------------
temperature_difference = (
    process_temp - air_temp
)

mechanical_load_proxy = (
    torque * rpm
)

torque_speed_ratio = (
    torque / (rpm + 1)
)


if st.button(
    "Predict Machine Failure",
    type="primary"
):

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Temperature Difference": [temperature_difference],
        "Mechanical Load Proxy": [mechanical_load_proxy],
        "Torque Speed Ratio": [torque_speed_ratio]
    })

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:

        if prediction == 1:
            st.error(
                "⚠️ Higher Risk: Machine failure predicted"
            )
        else:
            st.success(
                "✅ Lower Risk: Machine operating normally"
            )


    st.progress(
        min(float(probability), 1.0)
    )


    st.caption(
        f"Prediction generated using {model_name}."
    )


# -----------------------------
# Data table
# -----------------------------
st.divider()

st.subheader("📋 Filtered Machine Data")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Predictive Maintenance | Machine Learning Analytics Dashboard"
)
