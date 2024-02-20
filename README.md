# Fitness Tracker 
This project endeavors to leverage machine learning techniques to classify gym exercise types using data collected from devices equipped with accelerometers and gyroscopes. The overarching goal is to develop a robust classification model capable of accurately identifying various gym exercises based on patterns extracted from sensor data. By harnessing the rich information provided by accelerometers and gyroscopes, the project seeks to provide users with a reliable tool for automatically categorizing gym exercises, thereby streamlining workout tracking and analysis processes. Through the fusion of data science methodologies and sensor technology, the project aims to enhance the efficiency and effectiveness of gym workouts, ultimately promoting healthier lifestyles and fitness regimens.

# 1.	Chapter 1: Processing raw data  
## 1.1	Data Loading and Preprocessing Function:  
Reads accelerometer and gyroscope data from CSV files located in a specified directory.
Parses participant, label, and category information from file names.
Merges accelerometer and gyroscope data into separate dataframes.
Converts epoch timestamps to datetime and sets them as the index.
Drops unnecessary columns from the dataframes.  
## 1.2	Merging Datasets:  
Combines accelerometer and gyroscope data into a single dataframe.
Renames columns appropriately.  
## 1.3	Resampling Data:  
Resamples the merged data to a frequency (200ms).
Aggregates sensor readings within each resampled interval by taking the mean.
Splits the data into days before resampling and concatenates the results.
Converts the 'set' column to integer type.  
# 2.	Chapter 2: Detecting outliers in sensors  
## 2.1	Outlier Detection:  
### 2.1.1	Interquartile Range (IQR) Method:  
The function mark_outliers_iqr marks outliers in a specified column using the IQR method.  
### 2.1.2	Chauvenet's Criterion:  
The function mark_outliers_chauvenet identifies outliers based on Chauvenet's criterion, considering the deviation from the mean and standard deviation.  
### 2.1.3	Local Outlier Factor (LOF):  
The function mark_outliers_lof employs the Local Outlier Factor algorithm for outlier detection, determining outlier scores and marking outliers accordingly.  
## 2.2	Dealing with Outliers:  
### 2.2.1	Outlier Detection Method Selection:  
The script selects Chauvenet's criterion for outlier detection based on visual inspection.  
## 2.3	Outlier Treatment:  
Outliers identified by Chauvenet's criterion are replaced with NaNs in the dataset. The script iterates through each column and label, replacing outliers with NaNs in the original DataFrame.
# 3.	Chapter 3: Features engineering   
## 3.1	Dealing with Missing Values: 
Missing values in predictor columns are interpolated to ensure continuous data.  
## 3.2	Calculating Set Duration:  
 Duration of each set is calculated and categorized into heavy and medium repetition durations.  
## 3.3	Butterworth Lowpass Filter:  
A low-pass Butterworth filter is applied to remove high-frequency noise from the dataset, enhancing model accuracy by preserving underlying patterns while reducing noise effects.  
sf = 1000 / 200  # 200ms sampling frequency  
cutoff = 1.2 # 1.2Hz cutoff frequency after visual inspection  
 
## 3.4 Principal Component Analysis (PCA):  
PCA is employed to reduce data complexity by transforming variables into principal components, capturing most information with fewer variables, facilitating easier analysis and prediction.  
The choice of choosing the number of components principal was by using Elbow technique :  
 
Components principal plot:  

## 3.5 Sum of Squares Attributes:  
Scalar magnitudes (r) of accelerometer and gyroscope data are computed by summing squares of individual axes, enabling analysis independent of device orientation and dynamic re-orientations.  
### 3.5.1	Advantages:  
Device Orientation Independence: Using the scalar magnitude r makes the analysis independent of the device's orientation. Since it considers the total magnitude of the acceleration or angular velocity, it doesn't matter how the device is positioned in space.  
Dynamic Re-orientations: When a device undergoes dynamic movements or re-orientations, the individual axis measurements may change drastically. However, the scalar magnitude r encapsulates the overall intensity of the movement or rotation, making it suitable for handling such dynamic changes.  
### 3.5.2	Application: The scalar magnitude   
r can be used in various applications such as motion tracking, gesture recognition, activity monitoring, robotics, virtual reality, etc., where understanding the overall intensity of motion or rotation is more important than the specific direction.  
By calculating the scalar magnitude r, engineers and researchers can simplify the analysis of sensor data, making it more robust and applicable to a wide range of scenario.  
## 3.6	Temporal Abstraction:  
The first step involves temporal abstraction, where the data are transformed into a more manageable form for analysis. This is achieved by calculating statistical aggregates such as mean and standard deviation over rolling windows. Specifically, the mean and standard deviation are computed for accelerometer and gyroscope readings, facilitating smoother data analysis.  
  
 

## 3.7	Frequency Features Discrete Fourier Transformation (DFT):  
The Discrete Fourier Transformation (DFT) is a crucial tool for extracting frequency domain features from time-series data. By applying the DFT to preprocessed data, relevant frequency components are extracted, offering insights into the underlying patterns and trends present in the dataset. This is particularly valuable for machine learning applications, as it enables the representation of data in terms of frequency components, facilitating more efficient analysis and modeling. Moreover, the DFT aids in noise reduction, leading to more accurate models by filtering out irrelevant or noisy components from the data. Overall, the DFT enhances the understanding, modeling, and analysis of complex datasets, making it indispensable for tasks such as signal processing and pattern recognition.
Features we will be extracting:  
•	Amplitude (for each of the relevant frequencies that are part of the time window)  
•	Max frequency  
•	Weighted frequency (average)  
•	Power spectral entropy  
 
## 3.8	Dealing with Overlapping Windows:  
To mitigate potential overfitting issues and ensure robustness of the analysis, overlapping windows are handled appropriately. This involves dropping rows with missing values and subsampling the data to reduce redundancy and computational complexity, thereby enhancing the efficiency of subsequent analyses.  

## 3.9	Clustering:  
Clustering techniques are employed to group similar data points together based on their features. In this pipeline, k-means clustering is utilized to partition the data into distinct clusters. The optimal number of clusters is determined using the elbow method, and the clustering results are visualized to gain insights into the underlying patterns and structures within the data.  
Elbow method plot :  
 
Plot comparison between clusters and labels:  

# 4.	Chapter 4: training mode  
Checking distribution of training and test data:  

## 4.1	Split feature subsets:  
Different feature subsets are defined, including basic features, square features, PCA features, time features, frequency features, and cluster features.  
## 4.2	Forward Feature Selection:  
It employs forward feature selection method using a simple decision tree algorithm to select the most relevant features.  
The selected features and their corresponding scores are saved.  
## 4.3	Grid Search for Hyperparameters and Model Selection:  
It performs grid search to find the best hyperparameters for different classifiers (neural network, random forest, KNN, decision tree, naive Bayes) using various feature sets.  
The performance of each classifier with different feature sets is saved in a dataframe.  
## 4.4	Visualizing Model Performance:  
A grouped bar plot is created to compare the accuracy of different models with different feature sets.  
This provides insights into which feature set works best for each model.   

