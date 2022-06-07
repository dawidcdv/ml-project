import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from src.data.make_dataset import load_raw_train_data

# Load data
train = load_raw_train_data()
print(train.head())

# Standardize feature matrix
scaler = StandardScaler()
features_std = scaler.fit_transform(train)

# Caculate variance of each feature
thresholder_std = VarianceThreshold(threshold=.5)
features_high_variance_std = thresholder_std.fit_transform(features_std)
print(features_high_variance_std)

# Hight corelated features
corr_matrix = train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool_))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
df1 = train.drop(train.columns[to_drop], axis=1)
print(df1)