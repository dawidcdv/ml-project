import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from src.data.make_dataset import load_raw_train_data, load_labels
from src.features.helpers import absolute_path


def main():
    X = load_raw_train_data()
    y = load_labels()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_std)
    print(X_tsne.shape)

    sns.set(rc={'figure.figsize':(10,10)})
    palette = ['tab:red', 'tab:green']
    sns.scatterplot(data = X_tsne, x = X_tsne[:,0], y=X_tsne[:,1], hue=y[0], palette=palette)
    plt.savefig(absolute_path("reports","figures","tsne_scatter.png"))

    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(X_std)
    print('PCA n_components: ', pca.n_components_)

    pca = PCA(n_components=3558)
    X_pca = pca.fit_transform(X)

    tsne = TSNE()
    X_pca_tsne = tsne.fit_transform(X_pca)

    sns.scatterplot(data = X_pca_tsne, x = X_pca_tsne[:,0], y=X_pca_tsne[:,1], hue=y[0], palette=palette).set(title='TSNE + PCA')

    plt.savefig(absolute_path("reports","figures","tsne_pca_scatter.png"))


if __name__ == '__main__':
    main()
