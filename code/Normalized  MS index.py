# Normalize the initial PCA index (before limiting to 90% variance) to a 0-1 range
pca_index_initial_normalized = scaler.fit_transform(pca_index.reshape(-1, 1)).flatten()

# Convert to a pandas Series for plotting and analysis
pca_index_initial_normalized_series = pd.Series(pca_index_initial_normalized, index=consolidated_df.index, name="Normalized_Initial_PCA_Index")

# Plot the normalized initial PCA Index
plt.figure(figsize=(12, 6))
plt.plot(pca_index_initial_normalized_series, label='Normalized Initial PCA Index (0-1 Range)', color='purple')
plt.title("Normalized Initial PCA Index (0-1 Range) Over Time")
plt.xlabel("Date")
plt.ylabel("Normalized Initial PCA Index")
plt.legend()
plt.grid(True)
plt.show()

# Display the normalized initial PCA Index data to the user
tools.display_dataframe_to_user(name="Normalized Initial PCA Index (0-1 Range) Time Series", dataframe=pca_index_initial_normalized_series)
