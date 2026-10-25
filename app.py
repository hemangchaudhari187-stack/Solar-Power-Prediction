import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Solar AI | Power Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM UI / UX DESIGN
# ============================================================
st.markdown("""
<style>
:root {
    --green: #147A52;
    --green-dark: #10563D;
    --green-bright: #27966A;
    --ink: #20382D;
    --muted: #607568;
    --line: #DDE8E0;
    --canvas: #F5F8F5;
    --amber: #E5A23B;
}

.stApp {
    background: var(--canvas);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: rgba(245, 248, 245, 0.94);
}

[data-testid="stSidebar"] {
    background: #EAF2EC;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

[data-testid="stSidebar"] h1 {
    color: var(--green-dark) !important;
    font-size: 1.7rem;
    letter-spacing: -0.035em;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #244B38 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.46);
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 0.48rem 0.65rem;
    margin-bottom: 0.28rem;
    transition: background 0.15s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #FFFFFF;
    border-color: #C8DED0;
}

.hero {
    background: linear-gradient(
        120deg,
        #10563D 0%,
        #147A52 58%,
        #27966A 100%
    );
    padding: 2rem 2.1rem;
    border-radius: 20px;
    margin: 0.2rem 0 1.65rem 0;
    box-shadow: 0 10px 28px rgba(16, 86, 61, 0.12);
}

.hero,
.hero * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.hero h1 {
    font-size: clamp(1.8rem, 3vw, 2.45rem);
    letter-spacing: -0.045em;
    line-height: 1.15;
    margin: 0 0 0.65rem 0;
    font-weight: 800;
}

.hero p {
    color: #E8F5ED !important;
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
    max-width: 800px;
}

.eyebrow {
    color: #D2EBDD !important;
    font-size: 0.74rem;
    letter-spacing: 0.12em;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

.section-title {
    color: #173B2B !important;
    font-size: 1.22rem;
    font-weight: 780;
    letter-spacing: -0.02em;
    margin: 1.15rem 0 0.85rem 0;
}

.section-subtitle {
    color: #607568 !important;
    margin-top: -0.45rem;
    margin-bottom: 0.9rem;
    font-size: 0.92rem;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    padding: 1rem 1.15rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(22, 64, 43, 0.045);
    min-height: 108px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #5B7064 !important;
    opacity: 1 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: var(--green) !important;
    font-weight: 780;
    opacity: 1 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 16px !important;
    background: rgba(255, 255, 255, 0.82);
}

.stApp button[kind="primary"],
.stApp [data-testid="stFormSubmitButton"] button {
    background: var(--green) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--green) !important;
    border-radius: 10px;
    font-weight: 700;
    min-height: 2.7rem;
}

.stApp button[kind="primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover {
    background: var(--green-dark) !important;
    border-color: var(--green-dark) !important;
}

.result-card {
    background: linear-gradient(120deg, #E3F4E9, #F7FCF8);
    border: 1px solid #BBDCC8;
    border-radius: 18px;
    padding: 1.45rem 1.55rem;
    margin: 1rem 0 0.8rem 0;
    box-shadow: 0 6px 18px rgba(20, 108, 73, 0.06);
}

.result-label {
    color: #416653 !important;
    font-size: 0.98rem;
    font-weight: 650;
}

.result-value {
    color: #126842 !important;
    font-size: clamp(2rem, 4vw, 2.7rem);
    font-weight: 850;
    letter-spacing: -0.04em;
    margin-top: 0.25rem;
}

.small-note {
    color: #52685B !important;
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

.mini-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.05rem;
    height: 100%;
}

.mini-card h4 {
    color: #173B2B !important;
    font-size: 1rem;
    margin: 0 0 0.35rem 0;
}

.mini-card p {
    color: #607568 !important;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.5;
}

.status-pill {
    display: inline-block;
    padding: 0.35rem 0.65rem;
    border-radius: 99px;
    background: #DDF3E5;
    color: #126842 !important;
    font-weight: 650;
    font-size: 0.8rem;
}

[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] .stMarkdown strong,
[data-testid="stAppViewContainer"] [data-testid="stHeading"],
[data-testid="stAppViewContainer"] [data-testid="stHeading"] * {
    color: #243D30;
    opacity: 1;
}

[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] label p {
    color: #243D30 !important;
    opacity: 1 !important;
}

[data-testid="stAlert"],
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #173B2B !important;
    opacity: 1 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: var(--line);
}

#MainMenu,
footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .hero {
        padding: 1.35rem;
        border-radius: 15px;
    }

    .hero h1 {
        font-size: 1.75rem;
    }

    div[data-testid="stMetric"] {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY CHART STYLING
# ============================================================
def style_plotly_chart(fig, height=None):
    """Apply consistent, readable light-theme styling."""

    layout = {
        "template": "plotly_white",
        "paper_bgcolor": "#F5F8F5",
        "plot_bgcolor": "#FFFFFF",
        "font": {
            "family": "Arial, sans-serif",
            "size": 13,
            "color": "#243D30",
        },
        "title_font": {
            "color": "#173B2B",
            "size": 17,
        },
        "legend": {
            "font": {
                "color": "#243D30",
                "size": 12,
            },
            "bgcolor": "rgba(255,255,255,0)",
        },
        "margin": {
            "l": 22,
            "r": 20,
            "t": 58,
            "b": 28,
        },
    }

    if height is not None:
        layout["height"] = height

    fig.update_layout(**layout)

    fig.update_xaxes(
        title_font=dict(color="#243D30", size=13),
        tickfont=dict(color="#243D30", size=12),
        gridcolor="#E1E8E3",
        zerolinecolor="#CBD8CF",
        linecolor="#CBD8CF",
    )

    fig.update_yaxes(
        title_font=dict(color="#243D30", size=13),
        tickfont=dict(color="#243D30", size=12),
        gridcolor="#E1E8E3",
        zerolinecolor="#CBD8CF",
        linecolor="#CBD8CF",
    )

    return fig


def show_header(title, subtitle, eyebrow="SOLAR AI · POWER ANALYTICS"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD MODEL AND ANALYTICS
# Keep your existing filenames unchanged.
# ============================================================
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

    st.markdown(
        "Make sure these files are in the same folder as `app.py`: "
        "`solar_power_model.json`, `model_comparison.csv`, "
        "and `actual_vs_predicted.csv`."
    )

    st.exception(error)
    st.stop()


# Validate expected columns.
required_metrics = {"Model", "MAE", "RMSE", "R2"}
required_predictions = {
    "Actual_AC_Power",
    "Predicted_AC_Power",
}

if not required_metrics.issubset(metrics.columns):
    st.error(
        "model_comparison.csv is missing required columns: "
        "Model, MAE, RMSE, R2."
    )
    st.stop()

if not required_predictions.issubset(comparison.columns):
    st.error(
        "actual_vs_predicted.csv is missing required columns: "
        "Actual_AC_Power, Predicted_AC_Power."
    )
    st.stop()

if metrics.empty or comparison.empty:
    st.error("The analytics CSV files are empty.")
    st.stop()


# Ensure the metric and prediction columns are numeric.
for column in ["RMSE", "MAE", "R2"]:
    metrics[column] = pd.to_numeric(
        metrics[column],
        errors="coerce",
    )

for column in ["Actual_AC_Power", "Predicted_AC_Power"]:
    comparison[column] = pd.to_numeric(
        comparison[column],
        errors="coerce",
    )

metrics = metrics.dropna(subset=["RMSE", "MAE", "R2"])

comparison = comparison.dropna(
    subset=["Actual_AC_Power", "Predicted_AC_Power"]
).copy()

if metrics.empty or comparison.empty:
    st.error("The analytics files contain no usable numeric data.")
    st.stop()


best_row = metrics.loc[metrics["RMSE"].idxmin()]

model_colors = {
    "XGBoost": "#147A52",
    "Random Forest": "#64A984",
    "Linear Regression": "#E5A23B",
}


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
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
            "About",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**Project status**")

    st.markdown(
        '<span class="status-pill">● Model loaded</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown(
        '<span class="status-pill">● Analytics available</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Machine Learning · Solar Energy")
    st.caption("Educational project · Estimates only")


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================
if page == "Dashboard":

    show_header(
        "Solar Power Analytics",
        "Understand model performance and estimate solar plant AC "
        "power from weather measurements.",
    )

    section_title(
        "Performance overview",
        "A quick view of the model evaluation results.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best performing model",
            str(best_row["Model"]),
            help="Model with the lowest RMSE in the comparison table.",
        )

    with col2:
        st.metric(
            "Lowest RMSE",
            f'{best_row["RMSE"]:.3f}',
            help="Lower RMSE generally indicates smaller prediction errors.",
        )

    with col3:
        st.metric(
            "Best R² score",
            f'{best_row["R2"]:.4f}',
            help="R² describes how well the model explains variation "
                 "in the test data.",
        )

    section_title(
        "Actual vs predicted power",
        "Chronological test-set observations from the saved prediction results.",
    )

    sample = comparison.head(500).copy()
    sample["Observation"] = np.arange(1, len(sample) + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sample["Observation"],
            y=sample["Actual_AC_Power"],
            mode="lines",
            name="Actual",
            line=dict(color="#176B4D", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sample["Observation"],
            y=sample["Predicted_AC_Power"],
            mode="lines",
            name="Predicted",
            line=dict(color="#E5A23B", width=2),
        )
    )

    fig.update_layout(
        xaxis_title="Test observation",
        yaxis_title="AC power (dataset units)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    style_plotly_chart(fig, height=390)

    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.2, 1])

    with left:
        section_title("How to use Solar AI")

        st.markdown(
            "1. Open **Predict Power**.\n"
            "2. Enter ambient temperature, module temperature, "
            "and irradiation.\n"
            "3. Select **Predict Solar Power** to see the model estimate."
        )

    with right:
        st.info(
            "**Model limitation**\n\n"
            "This app estimates AC power from the weather measurements "
            "you enter. It does not forecast future weather or guarantee "
            "actual plant output."
        )


# ============================================================
# PAGE 2 — PREDICT POWER
# ============================================================
elif page == "Predict Power":

    show_header(
        "Solar Power Predictor",
        "Enter environmental conditions to estimate solar plant AC "
        "power using the trained XGBoost model.",
    )

    section_title(
        "Weather input parameters",
        "Enter values within the ranges below, then run the prediction.",
    )

    with st.container(border=True):

        with st.form("prediction_form"):

            col1, col2, col3 = st.columns(3)

            with col1:
                ambient = st.number_input(
                    "Ambient temperature (°C)",
                    min_value=-10.0,
                    max_value=60.0,
                    value=25.0,
                    step=1.0,
                    help="Temperature of the surrounding air.",
                )

            with col2:
                module = st.number_input(
                    "Module temperature (°C)",
                    min_value=-10.0,
                    max_value=100.0,
                    value=30.0,
                    step=1.0,
                    help="Temperature of the solar module surface.",
                )

            with col3:
                irradiation = st.number_input(
                    "Solar irradiation (dataset units)",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.05,
                    format="%.4f",
                    help="Use the same irradiation scale as the training dataset.",
                )

            submitted = st.form_submit_button(
                "☀️ Predict Solar Power",
                type="primary",
                width="stretch",
            )

    if submitted:

        input_df = pd.DataFrame(
            [{
                "AMBIENT_TEMPERATURE": ambient,
                "MODULE_TEMPERATURE": module,
                "IRRADIATION": irradiation,
            }]
        )

        # Preserve feature names and order used by the trained model.
        input_df = input_df[
            [
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION",
            ]
        ]

        try:
            prediction = max(
                0.0,
                float(model.predict(input_df)[0]),
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Estimated AC power output
                    </div>
                    <div class="result-value">
                        {prediction:,.2f}
                    </div>
                    <div class="small-note">
                        Prediction in the model's target units;
                        not a guaranteed real-world measurement.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            section_title("Conditions used for this estimate")

            a, b, c = st.columns(3)

            a.metric(
                "Ambient temperature",
                f"{ambient:.1f} °C",
            )

            b.metric(
                "Module temperature",
                f"{module:.1f} °C",
            )

            c.metric(
                "Irradiation",
                f"{irradiation:.4f}",
            )

            if irradiation <= 0.001:
                st.warning(
                    "Irradiation is nearly zero. Actual solar power "
                    "output would normally also be close to zero. "
                    "Treat the model estimate cautiously."
                )

            st.caption(
                "This estimate is generated by the saved XGBoost model "
                "using the values entered above."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Prediction workflow</h4>
                <p>
                    Enter three weather measurements and submit the form.
                    The saved model will estimate AC power using the
                    same feature names used during training.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    show_header(
        "Model Performance",
        "Compare machine-learning models using the same evaluation "
        "metrics and inspect the prediction errors.",
    )

    st.success(
        f'Lowest RMSE in the comparison: {best_row["Model"]} '
        f'({best_row["RMSE"]:.3f}).'
    )

    section_title(
        "Model comparison",
        "MAE and RMSE are error measures; R² describes model fit.",
    )

    display_metrics = metrics.copy()

    display_metrics["R2"] = display_metrics["R2"].round(4)
    display_metrics["MAE"] = display_metrics["MAE"].round(3)
    display_metrics["RMSE"] = display_metrics["RMSE"].round(3)

    st.dataframe(
        display_metrics,
        hide_index=True,
        width="stretch",
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_mae = px.bar(
            metrics,
            x="Model",
            y="MAE",
            title="Mean Absolute Error",
            color="Model",
            color_discrete_map=model_colors,
        )

        fig_mae.update_layout(
            showlegend=False,
            yaxis_title="MAE (target units)",
        )

        style_plotly_chart(fig_mae, height=350)

        st.plotly_chart(fig_mae, width="stretch")

    with col2:
        fig_rmse = px.bar(
            metrics,
            x="Model",
            y="RMSE",
            title="Root Mean Squared Error",
            color="Model",
            color_discrete_map=model_colors,
        )

        fig_rmse.update_layout(
            showlegend=False,
            yaxis_title="RMSE (target units)",
        )

        style_plotly_chart(fig_rmse, height=350)

        st.plotly_chart(fig_rmse, width="stretch")

    fig_r2 = px.bar(
        metrics,
        x="Model",
        y="R2",
        title="R² Score — Higher Is Generally Better",
        color="Model",
        color_discrete_map=model_colors,
    )

    fig_r2.update_layout(
        showlegend=False,
        yaxis_title="R²",
    )

    style_plotly_chart(fig_r2, height=350)

    st.plotly_chart(fig_r2, width="stretch")

    # Prediction error distribution — one heading, one chart.
    section_title(
        "Prediction error distribution",
        "Error is calculated as actual power minus predicted power.",
    )

    error_data = (
        comparison["Actual_AC_Power"]
        - comparison["Predicted_AC_Power"]
    ).dropna()

    if error_data.empty:
        st.warning(
            "No valid prediction errors are available in "
            "actual_vs_predicted.csv."
        )

    else:
        # Display central 98% to make the common errors easier to inspect.
        lower = float(error_data.quantile(0.01))
        upper = float(error_data.quantile(0.99))

        visible_errors = error_data[
            (error_data >= lower) & (error_data <= upper)
        ]

        outside_count = int(
            len(error_data) - len(visible_errors)
        )

        fig_error = px.histogram(
            x=visible_errors,
            nbins=50,
            title="Prediction errors — central 98% of observations",
            color_discrete_sequence=["#147A52"],
        )

        # Add the zero-error reference line before final styling.
        fig_error.add_vline(
            x=0,
            line_dash="dash",
            line_color="#D97706",
            annotation_text="Zero error",
            annotation_position="top",
        )

        fig_error.update_layout(
            xaxis_title="Actual power − predicted power (target units)",
            yaxis_title="Number of observations",
            showlegend=False,
        )

        style_plotly_chart(fig_error, height=370)

        # Render the histogram only once.
        st.plotly_chart(fig_error, width="stretch")

        st.caption(
            f"Showing {len(visible_errors):,} of "
            f"{len(error_data):,} observations. "
            f"{outside_count:,} extreme observations outside the "
            "displayed percentile range are excluded from this chart "
            "only, not from the model evaluation."
        )

    st.info(
        "Interpret MAE and RMSE in the target units of the dataset. "
        "R² should be considered alongside the error metrics and "
        "the dataset's limitations."
    )


# ============================================================
# PAGE 4 — ABOUT
# ============================================================
elif page == "About":

    show_header(
        "About Solar AI",
        "A machine-learning project for solar generation analytics, "
        "built to explore how weather conditions relate to estimated "
        "AC power output.",
    )

    section_title("Project overview")

    st.write(
        "Solar AI uses historical solar generation and weather "
        "measurements to estimate AC power output from selected "
        "environmental conditions. The app presents a trained XGBoost "
        "model alongside a comparison of three machine-learning approaches."
    )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("🌤️ Input features")

            st.markdown(
                "- Ambient temperature\n"
                "- Solar module temperature\n"
                "- Solar irradiation"
            )

    with right:
        with st.container(border=True):
            st.subheader("🤖 Models evaluated")

            st.markdown(
                "- Linear Regression\n"
                "- Random Forest\n"
                "- XGBoost"
            )

    section_title("How the prediction works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            <div class="mini-card">
                <h4>01 · Weather inputs</h4>
                <p>
                    Provide ambient temperature, module temperature,
                    and irradiation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="mini-card">
                <h4>02 · Trained model</h4>
                <p>
                    The saved XGBoost model processes the input features.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step3:
        st.markdown(
            """
            <div class="mini-card">
                <h4>03 · Power estimate</h4>
                <p>
                    The app displays estimated AC power output in the
                    model's target units.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("Evaluation metrics")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        "Mean Absolute Error",
    )

    m2.metric(
        "RMSE",
        "Root Mean Squared Error",
    )

    m3.metric(
        "R²",
        "Coefficient of Determination",
    )

    section_title("Limitations and responsible use")

    st.warning(
        "This project uses a limited historical dataset. Results are "
        "estimates, not guaranteed real-world output. The dashboard "
        "does not forecast future weather, and predictions should be "
        "validated before operational use."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "☀️ Solar AI Analytics · Educational Machine Learning Project · Estimates only"
)