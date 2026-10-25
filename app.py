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
    background: #F5F8F5;
    padding: 0.65rem 0.15rem 0.9rem 0.15rem;
    border-radius: 0;
    margin: 0.05rem 0 0.75rem 0;
    box-shadow: none;
    border-bottom: 1px solid #DDE8E0;
}

.hero,
.hero * {
    color: #20382D !important;
    opacity: 1 !important;
}

.hero h1 {
    font-size: clamp(1.45rem, 2.2vw, 1.9rem);
    letter-spacing: -0.045em;
    line-height: 1.15;
    margin: 0 0 0.65rem 0;
    font-weight: 800;
}

.hero p {
    color: #607568 !important;
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
    max-width: 800px;
}

.eyebrow {
    color: #147A52 !important;
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
        padding: 0.6rem 0.1rem;
        border-radius: 0;
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

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


def download_dataframe_button(df, label, file_name):
    csv_data = df.to_csv(index=False)
    return st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


def build_prediction_frame(ambient, module, irradiation):
    df = pd.DataFrame(
        [{
            "AMBIENT_TEMPERATURE": ambient,
            "MODULE_TEMPERATURE": module,
            "IRRADIATION": irradiation,
        }]
    )
    return df[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


def navigate_to_prediction():
    st.session_state["page"] = "Predict Power"


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
            "Accuracy & errors",
            "About",
        ],
        key="page",
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
        "A clearer view of solar performance.",
        "Monitor the trained solar-power model, review test-set results, and estimate AC power from weather conditions.",
    )

    st.button(
        "Predict AC power",
        key="dashboard_predict_button",
        on_click=navigate_to_prediction,
    )

    section_title(
        "Performance overview",
        "A quick view of the model evaluation results from the saved analytics files.",
    )

    daily_energy = float(comparison["Actual_AC_Power"].sum())
    predicted_daily_energy = float(comparison["Predicted_AC_Power"].sum())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Daily AC energy",
            f"{daily_energy:,.1f}",
            help="Total AC power across the available test-set observations.",
        )

    with c2:
        st.metric(
            "Predicted daily energy",
            f"{predicted_daily_energy:,.1f}",
            help="Total predicted AC power across the same observations.",
        )

    with c3:
        st.metric(
            "Test-set R²",
            f'{best_row["R2"]:.4f}',
            help="R² describes how well the model explains variation in the test data.",
        )

    with c4:
        st.metric(
            "MAE",
            f'{best_row["MAE"]:.3f}',
            help="Mean absolute error in the target units of the dataset.",
        )

    left, right = st.columns([1.7, 1])

    with left:
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

    with right:
        section_title("Solar plant conditions")
        with st.container(border=True):
            st.markdown(
                "<div class='mini-card'>"
                "<h4>Illustrative demo conditions</h4>"
                "<p>These values are not live plant telemetry; they are shown to demonstrate the dashboard layout and reading format.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.metric("Ambient temperature", "24.8 °C")
            st.metric("Module temperature", "31.3 °C")
            st.metric("Irradiation", "0.77")
            st.caption("Demo values only — no live facility telemetry is available in this project.")

    lower_left, lower_right = st.columns(2)

    with lower_left:
        section_title("Recent readings")
        st.dataframe(
            comparison.head(10)[["Actual_AC_Power", "Predicted_AC_Power"]].rename(
                columns={
                    "Actual_AC_Power": "Actual AC power",
                    "Predicted_AC_Power": "Predicted AC power",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with lower_right:
        section_title("Model insight")
        with st.container(border=True):
            st.markdown(
                f"<div class='mini-card'>"
                f"<h4>{best_row['Model']} leads</h4>"
                f"<p>Lowest RMSE: <strong>{best_row['RMSE']:.3f}</strong><br>"
                f"R²: <strong>{best_row['R2']:.4f}</strong><br>"
                f"MAE: <strong>{best_row['MAE']:.3f}</strong></p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption("Model comparison values are read from model_comparison.csv and update with the saved analytics file.")

    st.info(
        "This app estimates AC power from weather measurements. It does not forecast future weather or guarantee real plant output."
    )


# ============================================================
# PAGE 2 — PREDICT POWER
# ============================================================
elif page == "Predict Power":

    show_header(
        "Turn weather into a power estimate.",
        "Use the trained model to estimate AC power from ambient temperature, module temperature, and irradiation.",
    )

    prediction_defaults = {
        "prediction_ambient": 25.0,
        "prediction_module": 30.0,
        "prediction_irradiation": 0.5,
    }
    for key, value in prediction_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def load_prediction_example():
        st.session_state["prediction_ambient"] = 25.0
        st.session_state["prediction_module"] = 31.0
        st.session_state["prediction_irradiation"] = 0.75

    left, right = st.columns([1.15, 1.1])

    with left:
        section_title(
            "Weather input parameters",
            "Enter values within the ranges below, then run the prediction.",
        )

        with st.container(border=True):
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    ambient = st.number_input(
                        "AMBIENT_TEMPERATURE",
                        min_value=-10.0,
                        max_value=60.0,
                        step=1.0,
                        help="Ambient air temperature in °C.",
                        key="prediction_ambient",
                    )

                with col2:
                    module = st.number_input(
                        "MODULE_TEMPERATURE",
                        min_value=-10.0,
                        max_value=100.0,
                        step=1.0,
                        help="Solar module temperature in °C.",
                        key="prediction_module",
                    )

                with col3:
                    irradiation = st.number_input(
                        "IRRADIATION",
                        min_value=0.0,
                        max_value=2.0,
                        step=0.05,
                        format="%.4f",
                        help="Irradiation input in the same units used during model training.",
                        key="prediction_irradiation",
                    )

                submitted = st.form_submit_button(
                    "Predict AC power",
                    type="primary",
                    use_container_width=True,
                    key="prediction_submit",
                )

            st.button(
                "Load example",
                use_container_width=True,
                on_click=load_prediction_example,
                key="load_prediction_example",
            )

    with right:
        if submitted:
            input_df = build_prediction_frame(ambient, module, irradiation)
            try:
                prediction = max(0.0, float(model.predict(input_df)[0]))
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="result-label">Estimated AC power output</div>
                        <div class="result-value">""" + f"{prediction:,.2f}" + """</div>
                        <div class="small-note">Prediction in the model's target units; not a guaranteed real-world measurement.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div class='mini-card'><h4>Entered conditions</h4><p>Ambient: {0:.1f} °C<br>Module: {1:.1f} °C<br>Irradiation: {2:.4f}</p></div>".format(ambient, module, irradiation),
                    unsafe_allow_html=True,
                )

                if irradiation <= 0.001:
                    st.warning("Irradiation is nearly zero. Actual solar power output would normally also be close to zero; treat the estimate cautiously.")

                if irradiation > 0:
                    irradiations = np.linspace(max(0.0, irradiation * 0.5), min(1.5, irradiation * 1.5), 25)
                    what_if = []
                    for value in irradiations:
                        what_if.append(float(model.predict(build_prediction_frame(ambient, module, value))[0]))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=irradiations, y=what_if, mode="lines+markers", name="Predicted AC power", line=dict(color="#147A52", width=3)))
                    fig.update_layout(
                        title="How irradiation changes the estimate",
                        xaxis_title="Irradiation",
                        yaxis_title="Predicted AC power",
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    style_plotly_chart(fig, height=260)
                    st.plotly_chart(fig, use_container_width=True)

                st.caption("The estimate is generated by the saved XGBoost model using the feature order expected during training: AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION.")
            except Exception as error:
                st.error(f"Prediction failed: {error}")
        else:
            st.markdown(
                """
                <div class="mini-card">
                    <h4>Prediction workflow</h4>
                    <p>Enter three weather conditions and submit the form. The model will estimate AC power using the same features used during training.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("Valid inputs: ambient temperature and module temperature in °C; irradiation in the dataset's original units. This model estimates AC power and does not forecast future weather.")


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
    download_dataframe_button(display_metrics, "Download model comparison CSV", "model_comparison_export.csv")

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
# PAGE 4 — ACCURACY & ERRORS
# ============================================================
elif page == "Accuracy & errors":

    show_header(
        "Look beyond a single accuracy score",
        "Inspect the relationship between actual and predicted power, "
        "understand error distribution, and review the model's limitations.",
        eyebrow="SOLAR AI · MODEL DIAGNOSTICS",
    )

    # Use the XGBoost row when available; otherwise use the best-RMSE model.
    xgb_rows = metrics[metrics["Model"].astype(str).str.contains("xgboost", case=False)]
    selected_row = xgb_rows.iloc[0] if not xgb_rows.empty else best_row

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean absolute error", f'{selected_row["MAE"]:.3f}')
    c2.metric("Root mean squared error", f'{selected_row["RMSE"]:.3f}')
    c3.metric("R² score", f'{selected_row["R2"]:.4f}')
    error_values = comparison["Actual_AC_Power"] - comparison["Predicted_AC_Power"]
    c4.metric("Mean residual", f'{error_values.mean():.3f}')

    left, right = st.columns(2)
    with left:
        section_title("How close are the predictions?", "Points near the diagonal indicate closer agreement.")
        fig_scatter = px.scatter(
            comparison,
            x="Actual_AC_Power",
            y="Predicted_AC_Power",
            opacity=0.65,
            color_discrete_sequence=["#147A52"],
        )
        low = float(min(comparison["Actual_AC_Power"].min(), comparison["Predicted_AC_Power"].min()))
        high = float(max(comparison["Actual_AC_Power"].max(), comparison["Predicted_AC_Power"].max()))
        fig_scatter.add_trace(go.Scatter(
            x=[low, high], y=[low, high], mode="lines",
            name="Perfect prediction", line=dict(color="#E5A23B", dash="dash", width=2)
        ))
        fig_scatter.update_layout(
            xaxis_title="Actual AC power (dataset units)",
            yaxis_title="Predicted AC power (dataset units)",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )
        style_plotly_chart(fig_scatter, height=360)
        st.plotly_chart(fig_scatter, width="stretch")

    with right:
        section_title("Where do the errors fall?", "Residual = actual power − predicted power.")
        if error_values.empty:
            st.warning("No valid prediction errors are available.")
        else:
            lo, hi = float(error_values.quantile(0.01)), float(error_values.quantile(0.99))
            visible = error_values[(error_values >= lo) & (error_values <= hi)]
            fig_hist = px.histogram(
                x=visible, nbins=45,
                color_discrete_sequence=["#147A52"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#E5A23B")
            fig_hist.update_layout(
                xaxis_title="Actual − predicted (target units)",
                yaxis_title="Number of observations",
                showlegend=False,
            )
            style_plotly_chart(fig_hist, height=360)
            st.plotly_chart(fig_hist, width="stretch")
            st.caption(f"Showing central 98% of errors ({len(visible):,} of {len(error_values):,}); extreme values are hidden from this chart only.")

    error_records = comparison[["Actual_AC_Power", "Predicted_AC_Power"]].copy()
    error_records["Residual"] = error_records["Actual_AC_Power"] - error_records["Predicted_AC_Power"]
    download_dataframe_button(error_records, "Download error records CSV", "solar_power_error_records.csv")

    section_title("What to take away")
    st.markdown(
        "- **MAE** gives the average absolute prediction error in the target units.\\n"
        "- **RMSE** penalizes larger errors more heavily.\\n"
        "- **R²** measures how much variation is explained by the model; it is not a guarantee of real-world accuracy.\\n"
        "- Results depend on the quality and range of the historical data used to train and evaluate the model."
    )
    st.info("Use these diagnostics as a model-evaluation aid. Validate predictions on independent, representative data before operational use.")


# ============================================================
# PAGE 5 — ABOUT
# ============================================================
elif page == "About":

    show_header(
        "Built to explain. Designed to be questioned.",
        "This project packages a trained solar-power model, evaluation metrics, and interactive diagnostics into a compact analytics dashboard.",
    )

    st.markdown(
        "<div style='background:#EAF2EC;border:1px solid #DDE8E0;border-radius:16px;padding:1.2rem 1.3rem;margin:0 0 1.2rem 0;'>"
        "<strong style='color:#10563D'>Project purpose</strong><br>"
        "<span style='color:#20382D'>Solar AI estimates AC power from the weather measurements entered into the app. It is a model-exploration and evaluation interface, not a weather forecasting system.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    section_title("Project overview")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("🌤️ Input features and dataset coverage")
            st.markdown(
                "- Ambient temperature\n"
                "- Solar module temperature\n"
                "- Solar irradiation\n\n"
                "The app uses the same feature names and order as the trained XGBoost model. Feature values should be in the format used during training and evaluation."
            )

    with right:
        with st.container(border=True):
            st.subheader("🤖 Model and evaluation scope")
            st.markdown(
                "- Linear Regression\n"
                "- Random Forest\n"
                "- XGBoost\n\n"
                "The dashboard compares model quality using MAE, RMSE, and R² on the same evaluation set."
            )

    section_title("A simple prediction workflow")
    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("<div class='mini-card'><h4>01 · Prepare</h4><p>Collect the weather measurements needed by the model.</p></div>", unsafe_allow_html=True)
    with step2:
        st.markdown("<div class='mini-card'><h4>02 · Split</h4><p>Use the saved evaluation data to compare actual and predicted outputs.</p></div>", unsafe_allow_html=True)
    with step3:
        st.markdown("<div class='mini-card'><h4>03 · Train</h4><p>Fit the model on historical examples and preserve the trained artifact.</p></div>", unsafe_allow_html=True)
    with step4:
        st.markdown("<div class='mini-card'><h4>04 · Evaluate</h4><p>Review error metrics and residual patterns before drawing conclusions.</p></div>", unsafe_allow_html=True)

    section_title("Technology stack")
    tech = st.columns(5)
    labels = ["Python", "Streamlit", "Pandas", "XGBoost", "Plotly"]
    for col, label in zip(tech, labels):
        with col:
            st.markdown(f"<div class='mini-card'><h4>{label}</h4><p>Used in this project.</p></div>", unsafe_allow_html=True)

    section_title("Limitations and responsible use")
    st.warning(
        "This application estimates AC power from weather measurements entered by the user. It does not forecast future weather or guarantee real-world plant output, and any operational decision should be validated against representative observations."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "☀️ Solar AI Analytics · Educational Machine Learning Project · Estimates only"
)
