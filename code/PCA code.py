from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(consolidated_df)

# Step 2: Perform PCA
pca = PCA()
pca_components = pca.fit_transform(scaled_data)

# Calculate the PCA Index as a weighted sum of the principal components
explained_variance_ratio = pca.explained_variance_ratio_
pca_index = pca_components @ explained_variance_ratio  # Weighted sum of components

# Add the PCA Index to the original dataframe for time series indexing
pca_index_series = pd.Series(pca_index, index=consolidated_df.index, name="PCA_Index")

# Step 3: Plot the PCA Index over time
plt.figure(figsize=(12, 6))
plt.plot(pca_index_series, label='PCA Index', color='blue')
plt.title("PCA Index Over Time")
plt.xlabel("Date")
plt.ylabel("PCA Index")
plt.legend()
plt.grid(True)
plt.show()

# Display the final PCA Index series data to the user
tools.display_dataframe_to_user(name="PCA Index Time Series", dataframe=pca_index_series)
