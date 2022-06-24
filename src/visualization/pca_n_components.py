import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.data.make_dataset import load_raw_train_data, absolute_path
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    X = load_raw_train_data()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_std)
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    d = [n for n in range(len(cumsum))]

    plt.figure(figsize=(10, 10))
    plt.plot(d, cumsum, color='red', label='Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance as a Function of the Number of Components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Columns')
    plt.axhline(y=98, color='k', linestyle='--', label='98% Explained Variance')
    plt.axhline(y=95, color='yellow', linestyle='--', label='95% Explained Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(absolute_path("reports","figures","pca_n_components_scatter.png"))

if __name__ == '__main__':
    main()