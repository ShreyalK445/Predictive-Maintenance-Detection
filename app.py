import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)


# -----------------------------
# File locations
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "cleaned_predictive_maintenance.csv"
MODEL_FILE = BASE_DIR / "final_model.pkl"


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        st.error(
            "cleaned_predictive_maintenance.csv was not found "
            "in the GitHub repository."
        )
        st.stop()

    data = pd.read_csv(DATA_FILE)

    return data


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():
        st.error(
            "final_model.pkl was not found "
            "in the GitHub repository."
        )
        st.stop()

    package = joblib.load(MODEL_FILE)

    if isinstance(package, dict):

        trained_model = package["model"]

        model_name = package.get(
            "model_name",
            "Predictive Maintenance Model"
        )

        features = package.get(
            "features",
            []
        )

        threshold = package.get(
            "threshold",
            0.5
        )

    else:

        trained_model = package
        model_name = "Predictive Maintenance Model"
        features = []
        threshold = 0.5

    return (
        trained_model,
        model_name,
        features,
        threshold
    )


# -----------------------------
# Load everything
# -----------------------------
df = load_data()

model, model_name, model_features, threshold = load_model()


# -----------------------------
# Main title
# -----------------------------
st.title("⚙️ Predictive Maintenance Dashboard")

st.write(
    "Monitor machine conditions, explore failure patterns, "
    "and estimate machine failure risk using a trained ML model."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Filters")


# Machine type filter
type_values = sorted(
    df["Type"].dropna().unique().tolist()
)

type_filter = st.sidebar.selectbox(
    "Machine Type",
    ["All"] + type_values
)


# Machine status filter
status_filter = st.sidebar.selectbox(
    "Machine Status",
    ["All", "Normal", "Failure"]
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


if type_filter != "All":

    filtered_df = filtered_df[
        filtered_df["Type"] == type_filter
    ]


if status_filter == "Normal":

    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 0
    ]


elif status_filter == "Failure":

    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 1
    ]


# ============================================================
# KPI SECTION
# ============================================================

total_machines = len(filtered_df)

failure_count = int(
    filtered_df["Machine failure"].sum()
)

normal_count = (
    total_machines - failure_count
)

if total_machines > 0:

    failure_rate = (
        failure_count / total_machines
    ) * 100

else:

    failure_rate = 0


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Machines",
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


# ============================================================
# DASHBOARD CHARTS
# ============================================================

if len(filtered_df) > 0:

    st.subheader("📊 Failure Overview")

    chart_col1, chart_col2 = st.columns(2)


    # --------------------------------------------------------
    # Failure distribution
    # --------------------------------------------------------

    with chart_col1:

        failure_distribution = pd.DataFrame({

            "Status": [
                "Normal",
                "Failure"
            ],

            "Count": [
                normal_count,
                failure_count
            ]

        })

        fig = px.pie(
            failure_distribution,
            names="Status",
            values="Count",
            hole=0.45,
            title="Machine Failure Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Failure rate by machine type
    # --------------------------------------------------------

    with chart_col2:

        type_summary = (
            filtered_df
            .groupby("Type")["Machine failure"]
            .agg(
                Total="count",
                Failures="sum"
            )
            .reset_index()
        )

        type_summary["Failure Rate (%)"] = (
            type_summary["Failures"]
            / type_summary["Total"]
        ) * 100

        fig = px.bar(
            type_summary,
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


    # ========================================================
    # MACHINE CONDITION ANALYSIS
    # ========================================================

    st.subheader("🔍 Machine Condition Analysis")


    chart_col1, chart_col2 = st.columns(2)


    # Tool wear vs torque
    with chart_col1:

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


    # RPM vs torque
    with chart_col2:

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

    st.warning(
        "No machines match the selected filters."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")


model_col1, model_col2, model_col3 = st.columns(3)


model_col1.metric(
    "Model",
    model_name
)


model_col2.metric(
    "Decision Threshold",
    f"{threshold:.2f}"
)


model_col3.metric(
    "Input Features",
    len(model_features)
)


# ============================================================
# PREDICTION STUDIO
# ============================================================

st.divider()

st.subheader("🔧 Prediction Studio")

st.write(
    "Enter current machine readings to estimate the probability "
    "of machine failure."
)


input_col1, input_col2 = st.columns(2)


# ------------------------------------------------------------
# Left side inputs
# ------------------------------------------------------------

with input_col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"],
        key="prediction_type"
    )


    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )


    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=310.0,
        step=0.1
    )


    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0.0,
        max_value=5000.0,
        value=1500.0,
        step=10.0
    )


# ------------------------------------------------------------
# Right side inputs
# ------------------------------------------------------------

with input_col2:

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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

temperature_difference = (
    process_temperature
    - air_temperature
)


mechanical_load_proxy = (
    torque
    * rotational_speed
)


torque_speed_ratio = (
    torque
    / (rotational_speed + 1)
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    prediction_data = pd.DataFrame({

        "Type": [
            machine_type
        ],

        "Air temperature [K]": [
            air_temperature
        ],

        "Process temperature [K]": [
            process_temperature
        ],

        "Rotational speed [rpm]": [
            rotational_speed
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
            mechanical_load_proxy
        ],

        "Torque Speed Ratio": [
            torque_speed_ratio
        ]

    })


    try:

        failure_probability = (
            model
            .predict_proba(prediction_data)[0][1]
        )


        prediction = int(
            failure_probability >= threshold
        )


        st.subheader("Prediction Result")


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Failure Probability",
                f"{failure_probability * 100:.2f}%"
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
            min(
                max(
                    float(failure_probability),
                    0.0
                ),
                1.0
            )
        )


        st.write(
            f"Model used: **{model_name}**"
        )


    except Exception as error:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(error)


# ============================================================
# DATA TABLE
# ============================================================

st.divider()

st.subheader("📋 Machine Data")


st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictive Maintenance | Machine Learning Analytics Dashboard"
)
