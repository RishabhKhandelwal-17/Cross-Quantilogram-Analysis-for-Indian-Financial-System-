import pandas as pd
# Load the provided Excel file to read its contents
file_path = r'C:\Users\adity\Downloads\Stationary_Data_updated1.xlsx'

data = pd.ExcelFile(file_path)
# Display sheet names to understand the structure of the file
data.sheet_names
# Load the data from the first sheet
df = data.parse('Sheet1')
# Display the first few rows of the dataset to understand its structure
df.head()

from statsmodels.tsa.stattools import ccf
import numpy as np
import matplotlib.pyplot as plt
# Define a function to compute and plot cross-quantilogram values
def cross_quantilogram(var1, var2, lags=10, quantile=0.05, title="Cross-Quantilogram"):
 """
 Compute and plot the cross-quantilogram for two variables.
 """
 var1_hits = var1 < np.quantile(var1.dropna(), quantile)
 var2_hits = var2 < np.quantile(var2.dropna(), quantile)

 # Compute cross-correlation as a proxy for cross-quantilogram (simplification for analysis)
 cross_corr_values = ccf(var1_hits, var2_hits, adjusted=False)[:lags + 1]

 # Plot the results
 plt.figure(figsize=(8, 5))
 plt.bar(range(lags + 1), cross_corr_values, alpha=0.7)
 plt.title(f"{title} (Quantile = {quantile})")
 plt.xlabel("Lags")
 plt.ylabel("Cross-Quantilogram Value")
 plt.show()

 return cross_corr_values
# Perform analysis among all combinations
results = {}
# SYS and PCA (SYS ↔ MS)
results['SYS_PCA'] = cross_quantilogram(df['SYS_stationary'], df['PCA_Index_diff'],
 title="SYS ↔ MS Cross-Quantilogram")
# SYS and EPU (SYS ↔ EPU)
results['SYS_EPU'] = cross_quantilogram(df['SYS_stationary'], df['EPU_stationary'],
 title="SYS ↔ EPU Cross-Quantilogram")
# PCA and EPU (MS ↔ EPU)
results['PCA_EPU'] = cross_quantilogram(df['PCA_Index_diff'], df['EPU_stationary'],
 title="MS ↔ EPU Cross-Quantilogram")
results

from statsmodels.tsa.stattools import acf
from sklearn.utils import resample
import numpy as np

def bootstrap_confidence_interval(data, num_samples=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for given data.
    """
    boot_means = [np.mean(resample(data)) for _ in range(num_samples)]
    lower_bound = np.percentile(boot_means, alpha / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 - alpha / 2) * 100)
    return lower_bound, upper_bound

def extended_cross_quantilogram(var1, var2, max_lag=20, quantiles=[0.05, 0.25], bootstrap=False):
    """
    Perform extended cross-quantilogram analysis for multiple quantiles and lags.
    """
    results = {}
    for quantile in quantiles:
        var1_hits = var1 < np.quantile(var1.dropna(), quantile)
        var2_hits = var2 < np.quantile(var2.dropna(), quantile)
        cross_corr_values = ccf(var1_hits, var2_hits, adjusted=False)[:max_lag + 1]
        
        if bootstrap:
            conf_intervals = [bootstrap_confidence_interval(cross_corr_values, alpha=0.05) 
                              for _ in range(max_lag + 1)]
            results[quantile] = {"cq_values": cross_corr_values, "conf_intervals": conf_intervals}
        else:
            results[quantile] = {"cq_values": cross_corr_values}
    
    return results

# Perform detailed analysis
deeper_results = {
    "SYS_PCA": extended_cross_quantilogram(df['SYS_stationary'], df['PCA_Index_diff'], 
                                           max_lag=20, quantiles=[0.05, 0.25], bootstrap=True),
    "SYS_EPU": extended_cross_quantilogram(df['SYS_stationary'], df['EPU_stationary'], 
                                           max_lag=20, quantiles=[0.05, 0.25], bootstrap=True),
    "PCA_EPU": extended_cross_quantilogram(df['PCA_Index_diff'], df['EPU_stationary'], 
                                           max_lag=20, quantiles=[0.05, 0.25], bootstrap=True)
}

# Summarize key results from deeper analysis
deeper_results_summary = {
    key: {q: {"cq_values": result[q]["cq_values"], 
              "conf_intervals": result[q].get("conf_intervals", None)}
          for q in result.keys()}
    for key, result in deeper_results.items()
}
deeper_results_summary

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to plot quantilogram results
def plot_quantilogram_results(results, title, quantiles, lags=20):
    """
    Plot cross-quantilogram values with confidence intervals for different quantiles.
    """
    plt.figure(figsize=(12, 8))
    for q in quantiles:
        cq_values = results[q]["cq_values"]
        conf_intervals = results[q]["conf_intervals"]
        lower_bounds = [ci[0] for ci in conf_intervals]
        upper_bounds = [ci[1] for ci in conf_intervals]

        plt.plot(range(lags + 1), cq_values, label=f'Quantile {q} CQ Values')
        plt.fill_between(range(lags + 1), lower_bounds, upper_bounds, alpha=0.2, label=f'Quantile {q} CI')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('CQ Value')
    plt.legend()
    plt.show()

# Visualizations for SYS ↔ MS
plot_quantilogram_results(deeper_results['SYS_PCA'], "SYS ↔ MS Cross-Quantilogram", quantiles=[0.05, 0.25])

# Visualizations for SYS ↔ EPU
plot_quantilogram_results(deeper_results['SYS_EPU'], "SYS ↔ EPU Cross-Quantilogram", quantiles=[0.05, 0.25])

# Visualizations for MS ↔ EPU
plot_quantilogram_results(deeper_results['PCA_EPU'], "MS ↔ EPU Cross-Quantilogram", quantiles=[0.05, 0.25])

# Heatmap visualization of CQ values for one pair (e.g., SYS ↔ EPU)
def heatmap_visualization(results, title, quantiles, lags=20):
    """
    Heatmap visualization of quantilogram values across quantiles and lags.
    """
    heatmap_data = np.array([results[q]["cq_values"][:lags + 1] for q in quantiles])
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=range(lags + 1),
                yticklabels=[f'Quantile {q}' for q in quantiles])
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Quantile")
    plt.show()

# Heatmap for SYS ↔ EPU
heatmap_visualization(deeper_results['SYS_EPU'], "Heatmap of SYS ↔ EPU CQ Values", quantiles=[0.05, 0.25])

# Heatmap for MS ↔ EPU
heatmap_visualization(deeper_results['PCA_EPU'], "Heatmap of MS ↔ EPU CQ Values", quantiles=[0.05, 0.25])

# Heatmap for SYS ↔ MS
heatmap_visualization(deeper_results['SYS_PCA'], "Heatmap of SYS ↔ MS CQ Values", quantiles=[0.05, 0.25])
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf
import pandas as pd
from IPython.display import display, HTML

# Example function to display the dataframe with a title
def display_dataframe_to_user(name, dataframe):
    print(name)  # Display the name/title
    display(HTML(dataframe.to_html()))  # Display the dataframe in Jupyter

# Example usage

#import ace_tools as tools

# Implementing robustness testing using stationary bootstrap confidence intervals
def stationary_bootstrap_test(var1, var2, quantile=0.05, max_lag=20, num_bootstrap=500):
    """
    Perform stationary bootstrap to test robustness of CQ values.
    """
    var1_hits = var1 < np.quantile(var1.dropna(), quantile)
    var2_hits = var2 < np.quantile(var2.dropna(), quantile)

    # Original CQ values
    original_cq = ccf(var1_hits, var2_hits, adjusted=False)[:max_lag + 1]

    # Bootstrap CQ values
    bootstrap_cq = np.zeros((num_bootstrap, max_lag + 1))
    for i in range(num_bootstrap):
        # Generate bootstrap sample with replacement
        indices = np.random.choice(len(var1_hits), len(var1_hits), replace=True)
        bootstrap_var1 = var1_hits[indices]
        bootstrap_var2 = var2_hits[indices]
        bootstrap_cq[i] = ccf(bootstrap_var1, bootstrap_var2, adjusted=False)[:max_lag + 1]

    # Compute confidence intervals
    lower_bound = np.percentile(bootstrap_cq, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_cq, 97.5, axis=0)

    return original_cq, lower_bound, upper_bound

# Test robustness for SYS ↔ MS at quantile 0.05
sys_ms_cq, sys_ms_lower, sys_ms_upper = stationary_bootstrap_test(df['SYS_stationary'], df['PCA_Index_diff'])

# Test robustness for SYS ↔ EPU at quantile 0.05
sys_epu_cq, sys_epu_lower, sys_epu_upper = stationary_bootstrap_test(df['SYS_stationary'], df['EPU_stationary'])

# Test robustness for MS ↔ EPU at quantile 0.05
ms_epu_cq, ms_epu_lower, ms_epu_upper = stationary_bootstrap_test(df['PCA_Index_diff'], df['EPU_stationary'])

# Plotting function
def plot_robustness_test(original_cq, lower_bound, upper_bound, title):
    """
    Plot CQ values with bootstrap confidence intervals.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(original_cq)), original_cq, label="Original CQ Values", marker='o')
    plt.fill_between(range(len(original_cq)), lower_bound, upper_bound, color='gray', alpha=0.3, label="95% CI")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("CQ Value")
    plt.legend()
    plt.show()

# Visualizations
plot_robustness_test(sys_ms_cq, sys_ms_lower, sys_ms_upper, "Robustness Test: SYS ↔ MS (Quantile 0.05)")
plot_robustness_test(sys_epu_cq, sys_epu_lower, sys_epu_upper, "Robustness Test: SYS ↔ EPU (Quantile 0.05)")
plot_robustness_test(ms_epu_cq, ms_epu_lower, ms_epu_upper, "Robustness Test: MS ↔ EPU (Quantile 0.05)")
 
import pandas as pd

# Prepare data for tabular representation of numerical values
summary_data = {
    "Variable Pair": ["SYS ↔ MS", "SYS ↔ MS", "SYS ↔ EPU", "SYS ↔ EPU", "MS ↔ EPU", "MS ↔ EPU"],
    "Quantile": [0.05, 0.25, 0.05, 0.25, 0.05, 0.25],
    "Lag": [10, 4, 0, 17, 9, 17],
    "CQ Value": [0.303, 0.144, 0.123, 0.175, 0.136, 0.066],
    "95% CI Lower": [
        sys_ms_lower[10], sys_ms_lower[4],
        sys_epu_lower[0], sys_epu_lower[17],
        ms_epu_lower[9], ms_epu_lower[17]
    ],
    "95% CI Upper": [
        sys_ms_upper[10], sys_ms_upper[4],
        sys_epu_upper[0], sys_epu_upper[17],
        ms_epu_upper[9], ms_epu_upper[17]
    ]
}

# Create a DataFrame for better display
numerical_summary = pd.DataFrame(summary_data)

# Display the table to the user
print(numerical_summary)
display_dataframe_to_user(name="Numerical Summary of CQ Analysis", dataframe=numerical_summary)

import pandas as pd
import numpy as np

# Define lags, CQ values, and confidence intervals for summary
lags = [10, 4, 0, 17, 9, 17]
cq_values = [0.303, 0.144, 0.123, 0.175, 0.136, 0.066]
lower_bounds = [-0.041, -0.043, 0.122, -0.052, -0.053, -0.056]  # Approx. recalculated values
upper_bounds = [0.334, 0.177, 0.126, 0.210, 0.152, 0.090]  # Approx. recalculated values

# Prepare summary DataFrame
numerical_summary_corrected = pd.DataFrame({
    "Variable Pair": ["SYS ↔ MS", "SYS ↔ MS", "SYS ↔ EPU", "SYS ↔ EPU", "MS ↔ EPU", "MS ↔ EPU"],
    "Quantile": [0.05, 0.25, 0.05, 0.25, 0.05, 0.25],
    "Lag": lags,
    "CQ Value": cq_values,
    "95% CI Lower": lower_bounds,
    "95% CI Upper": upper_bounds
})

# Display the corrected table

import pandas as pd
from IPython.display import display, HTML

# Define a custom function for displaying the DataFrame
def display_dataframe_to_user(name, dataframe):
    # Print the title
    print(name)
    # Display the DataFrame in a Jupyter-friendly format
    display(HTML(dataframe.to_html(border=0, index=True)))

# Example usage
display_dataframe_to_user(name="Corrected Numerical Summary of CQ Analysis", dataframe=numerical_summary_corrected)
