import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "cleaned_predictive_maintenance.csv"
MODEL_FILE = BASE_DIR / "final_model.pkl"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error("Dataset file is missing from the GitHub repository.")
        st.stop()

    return pd.read_csv(DATA_FILE)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        st.error("Model file is missing from the GitHub repository.")
        st.stop()

    return joblib.load(MODEL_FILE)


df = load_data()
model_package = load_model()


# --------------------------------------------------
# GET MODEL INFORMATION
# --------------------------------------------------

if isinstance(model_package, dict):
    model = model_package["model"]
    model_name = model_package.get("model_name", "Saved Model")
    model_features = model_package.get("features", [])
else:
    model = model_package
    model_name = "Saved Model"
    model_features = []


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚙️ Predictive Maintenance Dashboard")

st.write(
    "Explore machine operating conditions and estimate the risk of machine failure."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

type_options = ["All"] + sorted(
    df["Type"].dropna().astype(str).unique().tolist()
)

selected_type = st.sidebar.selectbox(
    "Machine Type",
    type_options
)

failure_options = ["All", "Failure", "No Failure"]

selected_failure = st.sidebar.selectbox(
    "Machine Status",
    failure_options
)


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df.copy()

if selected_type != "All":
    filtered_df = filtered_df[
        filtered_df["Type"].astype(str) == selected_type
    ]

if selected_failure == "Failure":
    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 1
    ]

elif selected_failure == "No Failure":
    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 0
    ]


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

total_records = len(filtered_df)
failure_count = int(filtered_df["Machine failure"].sum())

if total_records > 0:
    failure_rate = (failure_count / total_records) * 100
    avg_tool_wear = filtered_df["Tool wear [min]"].mean()
else:
    failure_rate = 0
    avg_tool_wear = 0


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Records", f"{total_records:,}")

with col2:
    st.metric("Failures", f"{failure_count:,}")

with col3:
    st.metric("Failure Rate", f"{failure_rate:.2f}%")

with col4:
    st.metric("Avg Tool Wear", f"{avg_tool_wear:.1f} min")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.header("Machine Overview")

col1, col2 = st.columns(2)

with col1:

    failure_data = (
        filtered_df["Machine failure"]
        .value_counts()
        .rename(index={0: "No Failure", 1: "Failure"})
        .reset_index()
    )

    failure_data.columns = ["Status", "Count"]

    fig = px.bar(
        failure_data,
        x="Status",
        y="Count",
        title="Machine Failure Distribution",
        text="Count"
    )

    st.plotly_chart(fig, use_container_width=True)


with col2:

    type_failure = (
        filtered_df.groupby("Type")["Machine failure"]
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
        text="Failure Rate (%)"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# MACHINE CONDITIONS
# --------------------------------------------------

st.header("Machine Operating Conditions")

col1, col2 = st.columns(2)

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

    st.plotly_chart(fig, use_container_width=True)


with col2:

    fig = px.scatter(
        filtered_df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Machine failure",
        hover_data=["Type", "Tool wear [min]"],
        title="Rotational Speed vs Torque"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.header("Model Information")

st.info(
    f"Selected model: **{model_name}**"
)

st.write(
    "The model was trained using machine operating conditions and "
    "selected engineered features. The failure target is binary: "
    "0 = No Failure and 1 = Failure."
)


# --------------------------------------------------
# PREDICTION STUDIO
# --------------------------------------------------

st.header("🔧 Prediction Studio")

st.write(
    "Enter the current machine operating conditions to estimate "
    "the probability of machine failure."
)

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
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
        max_value=360.0,
        value=310.0,
        step=0.1
    )


with col2:

    rotational_speed = st.number_input(
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


# --------------------------------------------------
# FEATURE ENGINEERING FOR PREDICTION
# --------------------------------------------------

temperature_difference = (
    process_temperature - air_temperature
)

mechanical_load = (
    torque * rotational_speed
)

torque_speed_ratio = (
    torque / (rotational_speed + 1)
)


input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temperature],
    "Process temperature [K]": [process_temperature],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "Temperature Difference": [temperature_difference],
    "Mechanical Load Proxy": [mechanical_load],
    "Torque Speed Ratio": [torque_speed_ratio]
})


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        probability_percent = probability * 100

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"⚠️ Higher Failure Risk — "
                f"Estimated probability: {probability_percent:.2f}%"
            )

        else:

            st.success(
                f"✅ Lower Failure Risk — "
                f"Estimated probability: {probability_percent:.2f}%"
            )

        st.progress(float(probability))

        st.write(
            f"**Failure probability:** {probability_percent:.2f}%"
        )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.code(str(e))


# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("View Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Predictive Maintenance | Machine Learning Dashboard"
)
