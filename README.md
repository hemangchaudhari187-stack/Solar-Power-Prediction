# Solar AI — Solar Power Generation Prediction

## Overview
Solar AI is a machine-learning web application that estimates solar plant AC power output from weather measurements. It provides an interactive dashboard for predictions and model evaluation.

## Features
- Interactive dashboard with performance metrics
- Solar power prediction from weather inputs
- Comparison of Linear Regression, Random Forest, and XGBoost
- Visualizations of actual versus predicted power
- Prediction error distribution and evaluation charts

## Input Features
- Ambient temperature (°C)
- Module temperature (°C)
- Solar irradiation (dataset units)

## Best Model
XGBoost achieved the following results on the project's test dataset:
- MAE: 19.409
- RMSE: 53.549
- R²: 0.9786

These results describe the evaluation performed on this dataset and do not guarantee real-world performance.

## Technology Stack
- Python
- Streamlit
- Pandas and NumPy
- XGBoost
- Plotly
- Scikit-learn

## Run Locally
1. Clone this repository.
2. Install the dependencies:
   `pip install -r requirements.txt`
3. Start the application:
   `streamlit run app.py`

## Limitations
The model estimates power from the provided weather measurements. It does not forecast future weather. Results depend on the training dataset and should be validated before real-world use.
