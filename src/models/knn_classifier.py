import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.features.helpers import absolute_path
from src.data.make_dataset import load_labels, load_train_data


def get_knn_classifier(X_train, y_train, cache=True):
    if os.path.exists(KNN_FILE) and cache:
        knn = joblib.load(open(KNN_FILE, 'rb'))
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        joblib.dump(knn, KNN_FILE)

    return knn


def score_knn(X_train, y_train, X_test, y_test):
    knn = get_knn_classifier(X_train, y_train, False)
    y_pred = knn.predict(X_test)
    print("KNeighborsClassifier Accuracy:", accuracy_score(y_test, y_pred))


def main():
    X = load_train_data()
    y = load_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2022)
    score_knn(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    KNN_FILE = absolute_path('models', 'knn_classifier.pkl')
    main()
