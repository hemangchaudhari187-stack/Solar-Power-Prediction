
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Solar AI | Power Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9F7;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF1EC;
        border-right: 1px solid #DCE7DF;
    }

    [data-testid="stSidebar"] h1 {
        color: #146C49;
        font-size: 1.8rem;
    }

    .hero {
        background: linear-gradient(120deg, #125C40, #21865D);
        padding: 30px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #E1F3E8;
        margin: 0;
        font-size: 1rem;
    }

    .section-title {
        color: #173B2B;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE7DF;
        padding: 20px 20px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(20, 65, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #587064;
    }

    div[data-testid="stMetricValue"] {
        color: #146C49;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background-color: #16845B;
        border: none;
        border-radius: 9px;
        color: white;
        font-weight: 600;
        padding: 0.55rem 1.3rem;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #116744;
        color: white;
    }

    .result-card {
        background: linear-gradient(120deg, #E2F4E9, #F5FBF6);
        border: 1px solid #BBDCC8;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
    }

    .result-label {
        color: #416653;
        font-size: 1rem;
    }

    .result-value {
        color: #126842;
        font-size: 2.5rem;
        font-weight: 750;
        margin-top: 5px;
    }

   .small-note {
    color: #687D71;
    font-size: 0.88rem;
}

    #MainMenu, footer {
        visibility: hidden;
    }
    
/* Improve sidebar text visibility */
[data-testid="stSidebar"] {
    color: #173B2B !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] small {
    color: #173B2B !important;
}

/* Make navigation options readable */
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label * {
    color: #173B2B !important;
}

/* Make metric labels readable */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #587064 !important;
}

/* Keep primary buttons green */
.stApp button[kind="primary"],
.stApp [data-testid="stFormSubmitButton"] button {
    background-color: #16845B !important;
    color: #FFFFFF !important;
    border-color: #16845B !important;
}

/* Improve secondary text contrast */
.small-note,
[data-testid="stCaptionContainer"] {
    color: #52685B !important;
}

/* Make warning messages easy to read */
[data-testid="stAlert"] {
    color: #173B2B !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #173B2B !important;
}

/* Improve metric labels such as MAE, RMSE and R² */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] span {
    color: #52685B !important;
    opacity: 1 !important;
}

/* Keep metric values readable without truncating them too much */
div[data-testid="stMetricValue"] {
    color: #146C49 !important;
}

/* Fix faint text throughout the main content */
[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] .stMarkdown strong,
[data-testid="stAppViewContainer"] .stMarkdown h1,
[data-testid="stAppViewContainer"] .stMarkdown h2,
[data-testid="stAppViewContainer"] .stMarkdown h3 {
    color: #243D30 !important;
    opacity: 1 !important;
}

/* Fix labels and helper text inside input forms */
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] label p {
    color: #243D30 !important;
    opacity: 1 !important;
}

/* Ensure About-page lists are readable */
[data-testid="stAppViewContainer"] ul,
[data-testid="stAppViewContainer"] ol {
    color: #243D30 !important;
}

/* Keep secondary text readable */
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
    color: #52685B !important;
    opacity: 1 !important;
}

/* Fix headings rendered by Streamlit */
[data-testid="stHeading"],
[data-testid="stHeading"] *,
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #173B2B !important;
    opacity: 1 !important;
}

/* Keep headings inside bordered cards visible */
[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    color: #173B2B !important;
}

/* Consistent light background for chart containers */
[data-testid="stPlotlyChart"] {
    background-color: #F7F9F7 !important;
}


</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    saved_model = XGBRegressor()
    saved_model.load_model("solar_power_model.json")
    return saved_model

@st.cache_data
def load_analytics():
    model_metrics = pd.read_csv("model_comparison.csv")
    predictions = pd.read_csv("actual_vs_predicted.csv")
    return model_metrics, predictions

try:
    model = load_model()
    metrics, comparison = load_analytics()
except Exception as error:
    st.error("Could not load the model or analytics files.")
    st.caption(
        "Check that solar_power_model.json, model_comparison.csv, "
        "and actual_vs_predicted.csv are in the same folder as app.py."
    )
    st.exception(error)
    st.stop()

required_metrics = {"Model", "MAE", "RMSE", "R2"}
required_predictions = {"Actual_AC_Power", "Predicted_AC_Power"}

if not required_metrics.issubset(metrics.columns):
    st.error("The model comparison CSV is missing required columns.")
    st.stop()

if not required_predictions.issubset(comparison.columns):
    st.error("The prediction CSV is missing required columns.")
    st.stop()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.title("☀️ Solar AI")
    st.caption("RENEWABLE ENERGY ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Power",
            "Model Performance",
            "About"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**Project status**")
    st.markdown("🟢 Model loaded")
    st.markdown("📊 Analytics available")
    st.caption("Machine Learning • Solar Energy")

# --------------------------------------------------
# REUSABLE HEADER
# --------------------------------------------------
def show_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    best_row = metrics.loc[metrics["RMSE"].idxmin()]

    show_header(
        "Solar Power Analytics",
        "Understand model performance and estimate solar plant AC power."
    )

    st.markdown(
        '<div class="section-title">Performance Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Performing Model",
            str(best_row["Model"]),
            help="Model with the lowest RMSE in the comparison table."
        )

    with col2:
        st.metric(
            "Lowest RMSE",
            f'{best_row["RMSE"]:.2f}',
            help="Lower RMSE generally indicates smaller prediction errors."
        )

    with col3:
        st.metric(
            "Best R² Score",
            f'{best_row["R2"]:.4f}',
            help="R² describes how well the model explains variation in the test data."
        )

    st.markdown("")
    st.markdown(
        '<div class="section-title">Actual vs Predicted Power</div>',
        unsafe_allow_html=True
    )

    sample = comparison.head(500).copy()
    sample["Observation"] = np.arange(1, len(sample) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample["Observation"],
        y=sample["Actual_AC_Power"],
        mode="lines",
        name="Actual",
        line=dict(color="#176B4D", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=sample["Observation"],
        y=sample["Predicted_AC_Power"],
        mode="lines",
        name="Predicted",
        line=dict(color="#E5A23B", width=2)
    ))

    fig.update_layout(
        template="plotly_white",
        height=390,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_title="Test Observation",
        yaxis_title="AC Power (dataset units)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, width="stretch")

    st.info(
        "This model estimates power from the weather measurements you enter. "
        "It does not predict future weather or guarantee actual plant output."
    )

# --------------------------------------------------
# PREDICT POWER
# --------------------------------------------------
elif page == "Predict Power":
    show_header(
        "Solar Power Predictor",
        "Enter weather conditions to estimate solar plant AC power."
    )

    st.markdown(
        '<div class="section-title">Weather Input Parameters</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                ambient = st.number_input(
                    "Ambient Temperature (°C)",
                    min_value=-10.0,
                    max_value=60.0,
                    value=25.0,
                    step=1.0
                )

            with col2:
                module = st.number_input(
                    "Module Temperature (°C)",
                    min_value=-10.0,
                    max_value=100.0,
                    value=30.0,
                    step=1.0
                )

            with col3:
                irradiation = st.number_input(
                    "Solar Irradiation (dataset units)",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.05,
                    format="%.4f"
                )

            submitted = st.form_submit_button(
                "☀️ Predict Solar Power",
                type="primary",
                width="stretch"
            )

    if submitted:
        input_df = pd.DataFrame([{
            "AMBIENT_TEMPERATURE": ambient,
            "MODULE_TEMPERATURE": module,
            "IRRADIATION": irradiation
        }])

        try:
            prediction = max(
                0.0,
                float(model.predict(input_df)[0])
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated AC Power Output</div>
                    <div class="result-value">{prediction:,.2f}</div>
                    <div class="small-note">
                        Prediction in the model's target units.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            a, b, c = st.columns(3)
            a.metric("Ambient Temperature", f"{ambient:.1f} °C")
            b.metric("Module Temperature", f"{module:.1f} °C")
            c.metric("Irradiation", f"{irradiation:.4f}")

            if irradiation <= 0.001:
                st.warning(
                    "Irradiation is nearly zero. Actual solar power output "
                    "would normally also be close to zero. Treat this model "
                    "estimate cautiously."
                )

            st.caption(
                "This is a machine-learning estimate based on the trained "
                "model and the values entered above."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.caption(
            "Tip: Change the three input values, then select Predict Solar Power."
        )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    show_header(
        "Model Performance",
        "Compare machine-learning models using the same evaluation metrics."
    )

    best_row = metrics.loc[metrics["RMSE"].idxmin()]

    st.success(
        f"Lowest RMSE in the comparison: {best_row['Model']} "
        f"({best_row['RMSE']:.3f})."
    )

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    display_metrics = metrics.copy()
    display_metrics["R2"] = display_metrics["R2"].round(4)
    display_metrics["MAE"] = display_metrics["MAE"].round(3)
    display_metrics["RMSE"] = display_metrics["RMSE"].round(3)

    st.dataframe(
        display_metrics,
        hide_index=True,
        width="stretch"
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_mae = px.bar(
            metrics,
            x="Model",
            y="MAE",
            title="Mean Absolute Error",
            color="Model",
            color_discrete_sequence=["#16845B", "#64A984", "#E5A23B"]
        )
        fig_mae.update_layout(
            template="plotly_white",
            height=350,
            showlegend=False,
            yaxis_title="MAE (target units)"
        )
        st.plotly_chart(fig_mae, width="stretch")

    with col2:
        fig_rmse = px.bar(
            metrics,
            x="Model",
            y="RMSE",
            title="Root Mean Squared Error",
            color="Model",
            color_discrete_sequence=["#16845B", "#64A984", "#E5A23B"]
        )
        fig_rmse.update_layout(
            template="plotly_white",
            height=350,
            showlegend=False,
            yaxis_title="RMSE (target units)"
        )
        st.plotly_chart(fig_rmse, width="stretch")

    fig_r2 = px.bar(
        metrics,
        x="Model",
        y="R2",
        title="R² Score — Higher Is Generally Better",
        color="Model",
        color_discrete_sequence=["#16845B", "#64A984", "#E5A23B"]
    )
    fig_r2.update_layout(
        template="plotly_white",
        height=350,
        showlegend=False,
        yaxis_title="R²"
    )
    st.plotly_chart(fig_r2, width="stretch")

    st.markdown(
        '<div class="section-title">Prediction Error Distribution</div>',
        unsafe_allow_html=True
    )

    error_data = comparison.copy()
    error_data["Error"] = (
        error_data["Actual_AC_Power"]
        - error_data["Predicted_AC_Power"]
    )

    fig_error = px.histogram(
        error_data,
        x="Error",
        nbins=50,
        title="Actual Power − Predicted Power",
        color_discrete_sequence=["#16845B"]
    )
    fig_error.update_layout(
        template="plotly_white",
        height=370,
        xaxis_title="Prediction Error (target units)",
        yaxis_title="Number of observations"
    )
    st.plotly_chart(fig_error, width="stretch")

    st.caption(
        "MAE and RMSE are error measures, so lower values are generally better. "
        "R² measures fit and should be interpreted alongside error metrics."
    )

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About":
    show_header(
        "About Solar AI",
        "A machine-learning project for solar generation analytics."
    )

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Solar AI uses historical solar generation and weather measurements "
        "to estimate AC power output from selected environmental conditions."
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🌤️ Input Features")
            st.markdown("""
            - Ambient temperature
            - Solar module temperature
            - Solar irradiation
            """)

    with col2:
        with st.container(border=True):
            st.subheader("🤖 Models Evaluated")
            st.markdown("""
            - Linear Regression
            - Random Forest
            - XGBoost
            """)

    st.markdown(
        '<div class="section-title">Evaluation Metrics</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", "Mean Absolute Error")
    m2.metric("RMSE", "Root Mean Squared Error")
    m3.metric("R²", "Coefficient of Determination")

    st.warning(
        "This project uses a limited historical dataset. Results are estimates, "
        "not guaranteed real-world output. The dashboard does not forecast "
        "future weather, and its predictions should be validated before "
        "operational use."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "☀️ Solar AI Analytics | Educational Machine Learning Project"
)
