import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
window_size = 24  # 24-month rolling window
max_lag = 10

# Rolling window CQ for SYS ↔ MS at quantile 0.05
rolling_cq_sys_ms = rolling_window_cq(
    df['SYS_stationary'], 
    df['PCA_Index_diff'], 
    0.05, 
    window_size, 
    max_lag
)

# Rolling window CQ for SYS ↔ EPU at quantile 0.05
rolling_cq_sys_epu = rolling_window_cq(
    df['SYS_stationary'], 
    df['EPU_stationary'], 
    0.05, 
    window_size, 
    max_lag
)

# Rolling window CQ for MS ↔ EPU at quantile 0.05
rolling_cq_ms_epu = rolling_window_cq(
    df['PCA_Index_diff'], 
    df['EPU_stationary'], 
    0.05, 
    window_size, 
    max_lag
)

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
plot_rolling_cq(
    rolling_cq_sys_ms, 
    "Rolling Window CQ: SYS ↔ MS (Quantile 0.05)", 
    max_lag, 
    window_size - 1, 
    df['Dates']
)

# Rolling CQ plots for SYS ↔ EPU
plot_rolling_cq(
    rolling_cq_sys_epu, 
    "Rolling Window CQ: SYS ↔ EPU (Quantile 0.05)", 
    max_lag, 
    window_size - 1, 
    df['Dates']
)

# Rolling CQ plots for MS ↔ EPU
plot_rolling_cq(
    rolling_cq_ms_epu, 
    "Rolling Window CQ: MS ↔ EPU (Quantile 0.05)", 
    max_lag, 
    window_size - 1, 
    df['Dates']
)
