import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from src.data.make_dataset import load_raw_train_data, load_test_data, load_labels
from src.features.helpers import absolute_path
from scipy.stats import normaltest
from sklearn.decomposition import PCA


def nan_checking(train, test, labels):
    print('NaN in TRAIN')
    print(train.isna().sum().sort_values())
    print('NaN in TEST')
    print(test.isna().sum().sort_values())
    print('NaN in LABELS')
    print(labels.value_counts())
    print('LABELS countin')
    print(labels.isna().sum().sort_values())


def high_corelation(train):
    corr_matrix = train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print('High corelated coluns:', to_drop)


def variance(train):
    thresholder = VarianceThreshold(threshold=.5)
    train_high_variance = thresholder.fit_transform(train)
    print('Dataset shape after low variance removing', train_high_variance.shape)


def checking_norm_dist(train):
    is_normal = []

    for i in range(0, len(train.columns)):
        stats, p = normaltest(train[i])
        is_normal.append(1) if p > 0.05 else is_normal.append(0)

    print(pd.DataFrame(is_normal).value_counts())


def outliers_removing(train):
    def rm_sigma(dataFrame, column, sigma):
        mean = dataFrame[column].mean()
        std = dataFrame[column].std()
        sigma_thresh_up = mean + sigma * std
        sigma_thresh_down = mean - sigma * std 
        dataFrame = dataFrame[(dataFrame[column] < sigma_thresh_up) & (dataFrame[column] > sigma_thresh_down)]
        return dataFrame[column]

    sigma = 5

    df_clear = pd.DataFrame()
    for column in train.columns:
            df_clear = pd.concat([df_clear, rm_sigma(train, column, sigma)], axis=1)

    print(df_clear.isna().sum().sort_values())

    df_nan_rm = df_clear.dropna()

    print(df_nan_rm.isna().sum().sort_values())

    print(df_nan_rm.shape)


def pca_n_comp_calc(train):
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train)

    pca = PCA(n_components=0.95)
    pca.fit_transform(train_std)
    print('PCA n_components: ', pca.n_components_)


def main():
    train = load_raw_train_data()
    test = load_test_data()
    labels = load_labels()

    nan_checking(train, test, labels)
    high_corelation(train)
    variance(train)
    checking_norm_dist(train)
    outliers_removing(train)
    pca_n_comp_calc(train)


if __name__ == '__main__':
    main()