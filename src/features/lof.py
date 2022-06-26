from src.data.dataset import load_raw_train_data
from sklearn.neighbors import LocalOutlierFactor


def main():
    X = load_raw_train_data()
    for i in [15, 20, 25]:
        lof = LocalOutlierFactor(n_neighbors=i)
        y_pred = lof.fit_predict(X)
        print("N_neighbors: ", i)
        print("Lof outliers detected: ", y_pred[y_pred == -1])


if __name__ == '__main__':
    main()


