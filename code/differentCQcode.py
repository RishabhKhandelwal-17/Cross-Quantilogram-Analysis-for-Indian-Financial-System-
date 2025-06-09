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