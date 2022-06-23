import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from src.data.make_dataset import load_raw_train_data, load_labels
from src.features.helpers import absolute_path


X = load_raw_train_data()
y = load_labels()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

tsne = TSNE()
X_tsne = tsne.fit_transform(X_std)
print(X_tsne.shape)

