from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.data.dataset import load_raw_train_data, load_labels, load_test_data
from src.features.helpers import absolute_path
from src.models.svc import create_svc_classifier


def main():
    X = load_raw_train_data()
    y = load_labels()
    test_data = load_test_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    sc = StandardScaler(with_std=False)
    X_train = sc.fit_transform(X_train,y_train)

    classifier= create_svc_classifier()
    classifier.fit(X_train, y_train)
    predict_labels = classifier.predict(test_data)
    np.savetxt(absolute_path('data','processed','predictions.csv'), predict_labels, fmt='%i')


if __name__ == '__main__':
    main()
