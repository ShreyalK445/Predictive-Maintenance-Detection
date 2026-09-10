import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px


# ==============================
# Page Setup
# ==============================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "cleaned_predictive_maintenance.csv"
MODEL_FILE = BASE_DIR / "final_model.pkl"


# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        st.error(
            "Dataset file not found. "
            "Please upload cleaned_predictive_maintenance.csv "
            "to the GitHub repository."
        )
        st.stop()

    return pd.read_csv(DATA_FILE)


# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():
        st.error(
            "Model file not found. "
            "Please upload final_model.pkl "
            "to the GitHub repository."
        )
        st.stop()

    return joblib.load(MODEL_FILE)


df = load_data()
model_package = load_model()


# ==============================
# Model Information
# ==============================

if isinstance(model_package, dict):

    model = model_package["model"]

    model_name = model_package.get(
        "model_name",
        "Predictive Maintenance Model"
    )

    model_features = model_package.get(
        "features",
        []
    )

    threshold = model_package.get(
        "threshold",
        0.5
    )

else:

    model = model_package
    model_name = "Predictive Maintenance Model"
    model_features = []
    threshold = 0.5


# ==============================
# Title
# ==============================

st.title("⚙️ Predictive Maintenance Dashboard")

st.write(
    "Explore machine operating conditions and estimate "
    "the risk of machine failure."
)

st.divider()


# ==============================
# Sidebar Filters
# ==============================

st.sidebar.header("Dashboard Filters")

machine_types = ["All"] + sorted(
    df["Type"].dropna().astype(str).unique().tolist()
)

selected_type = st.sidebar.selectbox(
    "Machine Type",
    machine_types
)

status_options = [
    "All",
    "Failure",
    "No Failure"
]

selected_status = st.sidebar.selectbox(
    "Machine Status",
    status_options
)


# ==============================
# Apply Filters
# ==============================

filtered_df = df.copy()

if selected_type != "All":

    filtered_df = filtered_df[
        filtered_df["Type"].astype(str) == selected_type
    ]

if selected_status == "Failure":

    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 1
    ]

elif selected_status == "No Failure":

    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 0
    ]


# ==============================
# KPI Cards
# ==============================

total_records = len(filtered_df)

failure_count = int(
    filtered_df["Machine failure"].sum()
)

if total_records > 0:

    failure_rate = (
        failure_count / total_records
    ) * 100

    average_tool_wear = (
        filtered_df["Tool wear [min]"].mean()
    )

else:

    failure_rate = 0
    average_tool_wear = 0


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Records",
        f"{total_records:,}"
    )

with col2:
    st.metric(
        "Machine Failures",
        f"{failure_count:,}"
    )

with col3:
    st.metric(
        "Failure Rate",
        f"{failure_rate:.2f}%"
    )

with col4:
    st.metric(
        "Average Tool Wear",
        f"{average_tool_wear:.1f} min"
    )


# ==============================
# Machine Overview
# ==============================

st.header("Machine Overview")

col1, col2 = st.columns(2)


# Failure Distribution
with col1:

    failure_counts = (
        filtered_df["Machine failure"]
        .value_counts()
        .sort_index()
    )

    failure_chart = pd.DataFrame({
        "Status": [
            "No Failure",
            "Failure"
        ],
        "Count": [
            failure_counts.get(0, 0),
            failure_counts.get(1, 0)
        ]
    })

    fig = px.bar(
        failure_chart,
        x="Status",
        y="Count",
        text="Count",
        title="Machine Failure Distribution"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# Failure Rate by Type
with col2:

    type_data = (
        filtered_df
        .groupby("Type")["Machine failure"]
        .mean()
        .reset_index()
    )

    type_data["Failure Rate (%)"] = (
        type_data["Machine failure"] * 100
    )

    fig = px.bar(
        type_data,
        x="Type",
        y="Failure Rate (%)",
        text="Failure Rate (%)",
        title="Failure Rate by Machine Type"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==============================
# Operating Conditions
# ==============================

st.header("Machine Operating Conditions")

col1, col2 = st.columns(2)


# Tool Wear vs Torque
with col1:

    fig = px.scatter(
        filtered_df,
        x="Tool wear [min]",
        y="Torque [Nm]",
        color="Machine failure",
        hover_data=[
            "Type",
            "Rotational speed [rpm]",
            "Air temperature [K]",
            "Process temperature [K]"
        ],
        title="Tool Wear vs Torque"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# RPM vs Torque
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


# ==============================
# Temperature Analysis
# ==============================

st.subheader("Temperature Analysis")

temperature_df = filtered_df.copy()

temperature_df["Temperature Difference"] = (
    temperature_df["Process temperature [K]"]
    - temperature_df["Air temperature [K]"]
)

fig = px.scatter(
    temperature_df,
    x="Air temperature [K]",
    y="Process temperature [K]",
    color="Machine failure",
    hover_data=[
        "Type",
        "Temperature Difference"
    ],
    title="Air Temperature vs Process Temperature"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==============================
# Model Information
# ==============================

st.header("Machine Learning Model")

st.info(
    f"Final selected model: **{model_name}**"
)

st.write(
    "The model predicts whether a machine is likely to "
    "experience a failure based on its operating conditions."
)

if model_features:

    with st.expander("Features used by the model"):

        for feature in model_features:

            st.write(f"• {feature}")


# ==============================
# Prediction Studio
# ==============================

st.header("🔧 Prediction Studio")

st.write(
    "Enter the current machine operating conditions "
    "to estimate the failure risk."
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
        value=300.0,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=360.0,
        value=310.0,
        step=0.1
    )


with col2:

    rpm = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500.0,
        max_value=3000.0,
        value=1500.0,
        step=10.0
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=1.0
    )


# ==============================
# Feature Engineering
# ==============================

temperature_difference = (
    process_temp - air_temp
)

mechanical_load = (
    torque * rpm
)

torque_speed_ratio = (
    torque / (rpm + 1)
)


input_data = pd.DataFrame({

    "Type": [machine_type],

    "Air temperature [K]": [
        air_temp
    ],

    "Process temperature [K]": [
        process_temp
    ],

    "Rotational speed [rpm]": [
        rpm
    ],

    "Torque [Nm]": [
        torque
    ],

    "Tool wear [min]": [
        tool_wear
    ],

    "Temperature Difference": [
        temperature_difference
    ],

    "Mechanical Load Proxy": [
        mechanical_load
    ],

    "Torque Speed Ratio": [
        torque_speed_ratio
    ]
})


# ==============================
# Prediction
# ==============================

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    try:

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= threshold
        )

        probability_percent = (
            probability * 100
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"⚠️ Machine Failure Risk Detected\n\n"
                f"Estimated probability: "
                f"{probability_percent:.2f}%"
            )

        else:

            st.success(
                f"✅ No Immediate Failure Risk Detected\n\n"
                f"Estimated probability: "
                f"{probability_percent:.2f}%"
            )

        st.progress(
            float(probability)
        )

        st.write(
            f"**Failure probability:** "
            f"{probability_percent:.2f}%"
        )

        st.write(
            f"**Temperature Difference:** "
            f"{temperature_difference:.2f} K"
        )

        st.write(
            f"**Mechanical Load Proxy:** "
            f"{mechanical_load:.2f}"
        )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.code(
            str(e)
        )


# ==============================
# Dataset
# ==============================

st.header("Dataset")

with st.expander("View filtered dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# ==============================
# Footer
# ==============================

st.divider()

st.caption(
    "Predictive Maintenance Dashboard | Machine Learning Project"
)
