import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("data/interim/02_outliers_removed_chauvenets.pkl")
predictor_columns = list(df.columns[:6])

# plot setting

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for s in df["set"].unique():

    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]

    duration = end - start
    df.loc[df["set"] == s, "duration"] = duration.total_seconds()

duration_df = df.groupby("category")["duration"].mean()

heavy_repetition_duartion = duration_df.iloc[0] / 5  # 5 repetitions per heavy set
medium_repetition_duration = duration_df.iloc[1] / 10  # 10 repetitions per medium set

print("Heavy repetition duration: ", "{:.3f}".format(heavy_repetition_duartion))
print("Medium repetition duration: ", "{:.3f}".format(medium_repetition_duration))
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
"""A Butterworth low-pass filter is a type of filter that is used to remove high frequency noise 
from a dataset. It is most commonly used in machine learning in order to improve the accuracy of 
the model. The filter works by removing any data points above a certain threshold frequency, while 
still preserving the underlying pattern of the data. By doing so, it helps to reduce the effect of 
noise on the model, which can lead to better results.
"""

df_lowpass = df.copy()
LowPass = LowPassFilter()

sf = 1000 / 200  # 200ms sampling frequency}
cutoff = 1.2  # 1.2Hz cutoff frequency after visual inspection

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sf, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    df_lowpass.drop(columns=[col + "_lowpass"], inplace=True)

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
"""
PCA is a technique used in machine learning to reduce the complexity of data by transforming the data
into a new set of variables called principal components. This transformation is done in such a way that
the new set of variables captures the most amount of information from the original data set, while 
reducing the number of variables necessary. This helps to reduce the complexity of the data and make 
it easier to analyze and make predictions from.
"""
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pc_values) + 1), pc_values)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.show()

# 3 principal components was selected after visual inspection
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
"""
To further exploit the data, the scalar magnitudes r of the accelerometer and gyroscope 
were calculated. r is the scalar magnitude of the three combined data points: x, y, 
and z. The advantage of using r versus any particular data direction is that it is 
impartial to device orientation and can handle dynamic re-orientations.
"""
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)  # window size = 5

# the rolling function is used to calculate the mean and standard deviation of the data for each set
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features Discrete Fourier Transformation (DFT)
# --------------------------------------------------------------
"""
A DFT is beneficial for Machine Learning, as it can be used to represent data in terms of 
frequency components, allowing for more efficient analysis of the data. This provides a way 
to better understand and model complex data sets, as the frequency components produced by 
the DFT can provide insight into patterns and trends that would not otherwise be visible. 
Additionally, the DFT can be used to reduce noise, allowing for more accurate models.
"""

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

sf = 1000 / 200  # sampling frequency
ws = int(2800 / 200)  # windows size

df_freq_list = []

for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformation to set {s}")
    # get subset of data
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()

    # apply Fourier Transformation
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, sf)

    # append to list
    df_freq_list.append(subset)

# concatenate list of dataframes
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# drop rows with NaN values
df_freq = df_freq.dropna()

# get every second row to prevent overfitting by reducing the number of windows
df_freq = df_freq.iloc[::2]
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []
# Error : pip install threadpoolctl --upgrade  / upgrade to >= 3.x.x

# try different k values and calculate the sum of squared distances
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    # inertia is the sum of squared distances of samples to their closest cluster center
    inertias.append(kmeans.inertia_)

# plot the sum of squared distances for each k value
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("sum of squared distances")
plt.show()

# based on the elbow method, k=5 was selected
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# --------------------------------------------------------------
# Visualize comparison between clusters and labels

# visualize the clusters in 3D
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()

# visualize the labels in 3D
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("data/interim/03_data_features.pkl")
