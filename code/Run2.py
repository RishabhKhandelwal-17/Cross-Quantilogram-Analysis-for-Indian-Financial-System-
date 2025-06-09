# Load the dataset to start CQ analysis
import pandas as pd
from statsmodels.tsa.stattools import acf
from sklearn.utils import resample
import numpy as np
from IPython.display import display, HTML

file_path = r'C:\Users\adity\Downloads\Stationary_Data_updated1.xlsx'
data = pd.ExcelFile(file_path)
# Load the data from the first sheet
df = data.parse('Sheet1')
# Display the first few rows of the dataset to confirm structure
df.head()
# Handle missing values by forward filling, then backward filling as a precaution
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
# Ensure the 'Dates' column is in datetime format
df['Dates'] = pd.to_datetime(df['Dates'])
# Confirm preprocessing by displaying a summary of the dataset
df.info()
# Import necessary libraries for CQ computation and visualization
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
import seaborn as sns
# Define a function to compute CQ values across quantiles and lags
def compute_cq(var1, var2, quantile, max_lag):
 """
 Compute Cross-Quantilogram (CQ) for two variables.
 """
 var1_hits = var1 < np.quantile(var1, quantile)
 var2_hits = var2 < np.quantile(var2, quantile)
 return ccf(var1_hits, var2_hits, adjusted=False)[:max_lag + 1]
# Initialize parameters
quantiles = [0.05, 0.25]
max_lag = 20
# Compute CQ for SYS ↔ MS
cq_sys_ms = {q: compute_cq(df['SYS_stationary'], df['PCA_Index_diff'], q, max_lag) for q in
quantiles}
# Compute CQ for SYS ↔ EPU
cq_sys_epu = {q: compute_cq(df['SYS_stationary'], df['EPU_stationary'], q, max_lag) for q in
quantiles}
# Compute CQ for MS ↔ EPU
cq_ms_epu = {q: compute_cq(df['PCA_Index_diff'], df['EPU_stationary'], q, max_lag) for q in
quantiles}
# Visualize the CQ results
def plot_cq(cq_results, title):
    plt.figure(figsize=(10, 6))
    for q, cq_values in cq_results.items():
        plt.plot(range(len(cq_values)), cq_values, label=f'Quantile = {q}', marker='o')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('CQ Value')
    plt.legend()
    plt.show()
# Plot for SYS ↔ MS
plot_cq(cq_sys_ms, "Cross-Quantilogram: SYS ↔ MS")
# Plot for SYS ↔ EPU
plot_cq(cq_sys_epu, "Cross-Quantilogram: SYS ↔ EPU")
# Plot for MS ↔ EPU
plot_cq(cq_ms_epu, "Cross-Quantilogram: MS ↔ EPU")

# Create heatmaps for CQ values across quantiles and lags
def plot_cq_heatmap(cq_results, title, quantiles, max_lag):
 """
 Generate a heatmap of CQ values for different quantiles and lags.
 """
 heatmap_data = np.array([cq_results[q][:max_lag + 1] for q in quantiles])
 sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=range(max_lag + 1),
 yticklabels=[f'Quantile {q}' for q in quantiles])
 plt.title(title)
 plt.xlabel("Lag")
 plt.ylabel("Quantile")
 plt.show()
# Heatmap for SYS ↔ MS
plot_cq_heatmap(cq_sys_ms, "Heatmap: SYS ↔ MS", quantiles, max_lag)
# Heatmap for SYS ↔ EPU
plot_cq_heatmap(cq_sys_epu, "Heatmap: SYS ↔ EPU", quantiles, max_lag)
# Heatmap for MS ↔ EPU
plot_cq_heatmap(cq_ms_epu, "Heatmap: MS ↔ EPU", quantiles, max_lag)
# Prepare tabular summary of key CQ results for lags 0, 10, and 20
summary_data = []
key_lags = [0, 10, 20]
for var_pair, cq_results in zip(["SYS ↔ MS", "SYS ↔ EPU", "MS ↔ EPU"],
 [cq_sys_ms, cq_sys_epu, cq_ms_epu]):
 for q in quantiles:
    for lag in key_lags:
        summary_data.append([var_pair, q, lag, cq_results[q][lag]])
# Convert to DataFrame for display
summary_df = pd.DataFrame(summary_data, columns=["Variable Pair", "Quantile", "Lag", "CQ Value"])
# Display the tabular summary to the user
#tools.display_dataframe_to_user(name="Summary of CQ Results", dataframe=summary_df)
def display_dataframe_to_user(name, dataframe):
    """
    Display a DataFrame with a title in a Jupyter Notebook.
    """
    print(name)  # Print the title
    display(HTML(dataframe.to_html(index=False, border=0)))  # Display the DataFrame as HTML

# Display the summary
display_dataframe_to_user(name="Summary of CQ Results", dataframe=summary_df)
# Function for rolling window CQ analysis
def rolling_window_cq(var1, var2, quantile, window_size, max_lag):
 """
 Compute rolling window CQ values for two variables.
 """
 rolling_cq = []
 for i in range(len(var1) - window_size + 1):
 # Subset the rolling window
    window_var1 = var1[i:i + window_size]
    window_var2 = var2[i:i + window_size]

 # Compute CQ for the window
    cq_values = compute_cq(window_var1, window_var2, quantile, max_lag)
    rolling_cq.append(cq_values)

 return np.array(rolling_cq)
# Parameters for rolling window analysis
window_size = 24 # 24-month rolling window
max_lag = 10
# Rolling window CQ for SYS ↔ MS at quantile 0.05
rolling_cq_sys_ms = rolling_window_cq(df['SYS_stationary'], df['PCA_Index_diff'], 0.05, window_size, max_lag)
# Rolling window CQ for SYS ↔ EPU at quantile 0.05
rolling_cq_sys_epu = rolling_window_cq(df['SYS_stationary'], df['EPU_stationary'], 0.05, window_size, max_lag)
# Rolling window CQ for MS ↔ EPU at quantile 0.05
rolling_cq_ms_epu = rolling_window_cq(df['PCA_Index_diff'], df['EPU_stationary'], 0.05, window_size, max_lag)
# Visualization for rolling CQ
def plot_rolling_cq(rolling_cq, title, max_lag, start_date, dates):
 """
 Plot rolling CQ values for different lags.
 """
 plt.figure(figsize=(12, 8))
 for lag in range(max_lag + 1):
    plt.plot(dates[start_date:], rolling_cq[:, lag], label=f'Lag {lag}')
 plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
 plt.title(title)
 plt.xlabel('Date')
 plt.ylabel('CQ Value')
 plt.legend()
 plt.show()
# Rolling CQ plots for SYS ↔ MS
plot_rolling_cq(rolling_cq_sys_ms, "Rolling Window CQ: SYS ↔ MS (Quantile 0.05)", max_lag, window_size - 1, df['Dates'])
# Rolling CQ plots for SYS ↔ EPU
plot_rolling_cq(rolling_cq_sys_epu, "Rolling Window CQ: SYS ↔ EPU (Quantile 0.05)", max_lag,window_size - 1, df['Dates'])
# Rolling CQ plots for MS ↔ EPU
plot_rolling_cq(rolling_cq_ms_epu, "Rolling Window CQ: MS ↔ EPU (Quantile 0.05)", max_lag, window_size - 1, df['Dates'])
# Lag-specific dependency trends: Plot CQ values for a specific lag across all quantiles
def plot_lag_dependency_trends(cq_results, lag, quantiles, title):
 """
 Plot CQ values for a specific lag across all quantiles.
 """
 lag_values = [cq_results[q][lag] for q in quantiles]
 plt.figure(figsize=(8, 5))
 plt.plot(quantiles, lag_values, marker='o', label=f'Lag {lag} Dependency')
 plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
 plt.title(title)
 plt.xlabel("Quantiles")
 plt.ylabel("CQ Value")
 plt.legend()
 plt.show()
# Lag-specific trends for lag 10 across all quantiles
plot_lag_dependency_trends(cq_sys_ms, 10, quantiles, "Lag 10 Dependency: SYS ↔ MS")
plot_lag_dependency_trends(cq_sys_epu, 10, quantiles, "Lag 10 Dependency: SYS ↔ EPU")
plot_lag_dependency_trends(cq_ms_epu, 10, quantiles, "Lag 10 Dependency: MS ↔ EPU")
# Comparison bar plots: Compare CQ values across variable pairs for specific quantiles and lags
def plot_comparison_bar(cq_results_list, var_pairs, quantiles, lag, title):
 """
 Compare CQ values across variable pairs for specific quantiles and lag.
 """
 bar_values = []
 for cq_results in cq_results_list:
    bar_values.append([cq_results[q][lag] for q in quantiles])

 bar_values = np.array(bar_values)
 bar_width = 0.2
 x = np.arange(len(quantiles))

 plt.figure(figsize=(10, 6))
 for i, (var_pair, values) in enumerate(zip(var_pairs, bar_values)):
    plt.bar(x + i * bar_width, values, bar_width, label=var_pair)

 plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
 plt.xticks(x + bar_width, [f"Quantile {q}" for q in quantiles])
 plt.title(title)
 plt.xlabel("Quantiles")
 plt.ylabel("CQ Value")
 plt.legend()
 plt.show()
# Comparison bar plots for lag 10
plot_comparison_bar(
 [cq_sys_ms, cq_sys_epu, cq_ms_epu],
 ["SYS ↔ MS", "SYS ↔ EPU", "MS ↔ EPU"],
 quantiles,
 10,
 "Comparison of CQ Values at Lag 10"
)
# Lag-specific trends for a different lag (e.g., Lag 5) across all quantiles
plot_lag_dependency_trends(cq_sys_ms, 5, quantiles, "Lag 5 Dependency: SYS ↔ MS")
plot_lag_dependency_trends(cq_sys_epu, 5, quantiles, "Lag 5 Dependency: SYS ↔ EPU")
plot_lag_dependency_trends(cq_ms_epu, 5, quantiles, "Lag 5 Dependency: MS ↔ EPU")
# Comparison bar plots for a different lag (e.g., Lag 5)
plot_comparison_bar(
 [cq_sys_ms, cq_sys_epu, cq_ms_epu],
 ["SYS ↔ MS", "SYS ↔ EPU", "MS ↔ EPU"],
 quantiles,
 5,
 "Comparison of CQ Values at Lag 5"
)



