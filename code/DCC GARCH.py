import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load data
data = pd.read_excel("financial_data.xlsx", parse_dates=True, index_col="Date")
returns = np.log(data / data.shift(1)).dropna()

# Dictionary to store model parameters
garch_parameters = {}

# Fit GARCH(1,1) model for each institution and print parameters
for column in returns.columns:
    garch = arch_model(returns[column], vol='Garch', p=1, q=1)
    fit = garch.fit(disp="off")
    garch_parameters[column] = fit.params  # Save parameters for each institution
    
    # Print the model parameters
    print(f"GARCH Model Parameters for {column}:")
    print(fit.params)
    print("\n" + "="*40 + "\n")

# Convert parameters to a DataFrame for easier comparison
garch_params_df = pd.DataFrame(garch_parameters)
print("GARCH Model Parameters for All Institutions:")
print(garch_params_df)

# Fit univariate GARCH(1,1) models
garch_models = {}
residuals = pd.DataFrame(index=returns.index)

for column in returns.columns:
    garch = arch_model(returns[column], vol='Garch', p=1, q=1)
    fit = garch.fit(disp="off")
    residuals[column] = fit.resid / fit.conditional_volatility  # Standardized residuals
    garch_models[column] = fit

# Calculate dynamic correlations manually
rolling_window = 20  # Adjust the window size as needed
correlations = pd.DataFrame(index=returns.index[rolling_window - 1:])

for i in range(len(returns.columns)):
    for j in range(i + 1, len(returns.columns)):
        pair = (returns.columns[i], returns.columns[j])
        correlations[f"{pair[0]}-{pair[1]}"] = residuals[returns.columns[i]].rolling(rolling_window).corr(residuals[returns.columns[j]])

# Calculate the systemic risk index as the average correlation at each time point
sys_index = correlations.mean(axis=1)


# Sys Index Plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sys_index.index,
    y=sys_index.values,
    mode="lines",
    name="Systemic Risk Index (Sys)",
    line=dict(color="blue")
))

fig.update_layout(
    title="Systemic Risk Index Over Time",
    xaxis_title="Date",
    yaxis_title="Systemic Risk Index (Sys)",
    template="plotly_white"
)

fig.show()

# Tabular output of Sys index
sys_index_df = pd.DataFrame({"Date": sys_index.index, "Systemic Risk Index (Sys)": sys_index.values})
print(sys_index_df)