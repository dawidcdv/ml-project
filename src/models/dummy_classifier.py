import os
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.features.helpers import absolute_path
from src.data.make_dataset import load_raw_train_data, load_labels


def get_dummy_classifier(X_train, y_train, cache=True):
    if os.path.exists(DC_DUMP_FILE) and cache:
        dummy = joblib.load(open(DC_DUMP_FILE, 'rb'))
    else:
        dummy = DummyClassifier()
        dummy.fit(X_train, y_train)
        joblib.dump(dummy, DC_DUMP_FILE)

    return dummy


def score_dummy(X_train, y_train, X_test, y_test):
    dummy = get_dummy_classifier(X_train, y_train, False)
    y_pred = dummy.predict(X_test)
    print("DummyClassifier Accuracy:", accuracy_score(y_test, y_pred))


def main():
    X = load_raw_train_data()
    y = load_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2022)
    score_dummy(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    DC_DUMP_FILE = absolute_path('models', 'dummy_classifier.pkl')
    main()
