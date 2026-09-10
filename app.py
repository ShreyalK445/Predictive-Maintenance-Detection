import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# -----------------------------
# PAGE SETUP
# -----------------------------

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance Dashboard")
st.write(
    "Explore machine operating conditions and estimate "
    "the risk of machine failure."
)


# -----------------------------
# LOAD DATA AND MODEL
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "cleaned_predictive_maintenance.csv"
    )


@st.cache_resource
def load_model():
    return joblib.load(
        "final_model.pkl"
    )


df = load_data()
model_package = load_model()

model = model_package["model"]
features = model_package["features"]
model_name = model_package["model_name"]


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Filters")

machine_types = st.sidebar.multiselect(
    "Machine Type",
    options=sorted(df["Type"].dropna().unique()),
    default=sorted(df["Type"].dropna().unique())
)

failure_filter = st.sidebar.selectbox(
    "Failure Status",
    ["All", "Failure", "No Failure"]
)


filtered_df = df[
    df["Type"].isin(machine_types)
].copy()


if failure_filter == "Failure":
    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 1
    ]

elif failure_filter == "No Failure":
    filtered_df = filtered_df[
        filtered_df["Machine failure"] == 0
    ]


# -----------------------------
# KPI SECTION
# -----------------------------

st.subheader("Overview")

total_records = len(filtered_df)

total_failures = filtered_df[
    "Machine failure"
].sum()

failure_rate = (
    total_failures / total_records * 100
    if total_records > 0 else 0
)

avg_tool_wear = filtered_df[
    "Tool wear [min]"
].mean()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Records",
    f"{total_records:,}"
)

col2.metric(
    "Failures",
    f"{total_failures:,}"
)

col3.metric(
    "Failure Rate",
    f"{failure_rate:.2f}%"
)

col4.metric(
    "Average Tool Wear",
    f"{avg_tool_wear:.1f} min"
)


# -----------------------------
# EDA
# -----------------------------

st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)


with col1:

    failure_counts = (
        filtered_df["Machine failure"]
        .value_counts()
        .rename(
            index={
                0: "No Failure",
                1: "Failure"
            }
        )
        .reset_index()
    )

    failure_counts.columns = [
        "Status",
        "Count"
    ]

    fig = px.bar(
        failure_counts,
        x="Status",
        y="Count",
        title="Machine Failure Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    type_failure = (
        filtered_df
        .groupby("Type")["Machine failure"]
        .mean()
        .mul(100)
        .reset_index()
    )

    type_failure.columns = [
        "Type",
        "Failure Rate (%)"
    ]

    fig = px.bar(
        type_failure,
        x="Type",
        y="Failure Rate (%)",
        title="Failure Rate by Machine Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -----------------------------
# SENSOR ANALYSIS
# -----------------------------

st.subheader("Sensor Analysis")

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


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.subheader("Model Performance")

st.info(
    f"Selected model: {model_name}"
)

st.write(
    "The model was selected using F1-score during "
    "cross-validation on the training data and then "
    "evaluated on unseen test data."
)


# -----------------------------
# PREDICTION STUDIO
# -----------------------------

st.subheader("🔮 Machine Failure Prediction")

st.write(
    "Enter the current machine operating conditions."
)

col1, col2 = st.columns(2)


with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        value=310.0
    )

    rpm = st.number_input(
        "Rotational Speed [rpm]",
        value=1500.0
    )


with col2:

    torque = st.number_input(
        "Torque [Nm]",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        value=100.0
    )


# Feature engineering for prediction

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


if st.button(
    "Predict Machine Failure",
    type="primary"
):

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    st.divider()

    if prediction == 1:

        st.error(
            f"⚠️ Higher Failure Risk — "
            f"Probability: {probability * 100:.2f}%"
        )

    else:

        st.success(
            f"✓ Lower Failure Risk — "
            f"Probability: {probability * 100:.2f}%"
        )


# -----------------------------
# RAW DATA
# -----------------------------

with st.expander("View Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


st.caption(
    "Predictive maintenance analysis developed for ML NEURONETS 2.0."
)
