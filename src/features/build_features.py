import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from src.data.make_dataset import load_raw_train_data, load_test_data, load_labels
from src.features.helpers import absolute_path
from scipy.stats import normaltest

# Load data
train = load_raw_train_data()
test = load_test_data()
labels = load_labels()

# NaN checking
print('NaN in TRAIN')
print(train.isna().sum().sort_values())
print('NaN in TEST')
print(test.isna().sum().sort_values())
print('NaN in LABELS')
print(labels.value_counts())
print('LABELS countin')
print(labels.isna().sum().sort_values())

# High corelation
corr_matrix = train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool_))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print('High corelated coluns:', to_drop)

# Caculate variance
thresholder = VarianceThreshold(threshold=.5)
train_high_variance = thresholder.fit_transform(train)
print('Dataset shape after low variance removing', train_high_variance.shape)

# Checking normal distribution
is_normal = []

for i in range(0, len(train.columns)):
    stats, p = normaltest(train[i])
    is_normal.append(1) if p > 0.05 else is_normal.append(0)

print(pd.DataFrame(is_normal).value_counts())

# Outliers removing
def rm_sigma(dataFrame, column, sigma):
    # Calculate mean for data column
    mean = dataFrame[column].mean()
    # Calculate std 
    std = dataFrame[column].std()
    # Define a thresholds
    sigma_thresh_up = mean + sigma * std
    sigma_thresh_down = mean - sigma * std    
    # Remove an outlier data
    dataFrame = dataFrame[(dataFrame[column] < sigma_thresh_up) & (dataFrame[column] > sigma_thresh_down)]
    return dataFrame[column]

sigma = 3

df_clear = pd.DataFrame()
for column in train.columns:
        df_clear = pd.concat([df_clear, rm_sigma(train, column, sigma)], axis=1)

print(df_clear.isna().sum().sort_values())

df_nan_rm = df_clear.dropna()

print(df_nan_rm.isna().sum().sort_values())

print(df_nan_rm.shape)

# # Standardize feature matrix
scaler = StandardScaler()
train_std = scaler.fit_transform(train)

# Save data after cleaning
train_std.to_csv(absolute_path("data","processed","train_data.csv"), header=False)