import argparse
import os
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from src.features.helpers import absolute_path
from src.data.make_dataset import load_labels, load_raw_train_data
from src.models.train_helper import verify_model


def get_args():
    parser = argparse.ArgumentParser("Dummy Classifier - Baseline model")
    parser.add_argument("--cache", help="Load model from cache if exist", type=bool, default=False)
    return parser.parse_args()


def get_dummy_classifier(X_train, y_train, cache=False):
    if os.path.exists(DC_DUMP_FILE) and cache:
        dummy = joblib.load(open(DC_DUMP_FILE, 'rb'))
    else:
        dummy = DummyClassifier( )
        dummy.fit(X_train, y_train)
        joblib.dump(dummy, DC_DUMP_FILE)

    return dummy


def score_dummy(X_train, y_train, X_test, y_test, cache=False):
    dummy = get_dummy_classifier(X_train, y_train, cache)
    verify_model(dummy, X_train, y_train, X_test, y_test, absolute_path("reports", "figures", "dummy_cm.jpg"))


def main():
    args = get_args()
    X = load_raw_train_data()
    y = load_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    score_dummy(X_train, y_train, X_test, y_test, args.cache)


if __name__ == '__main__':
    DC_DUMP_FILE = absolute_path('models', 'dummy_classifier.pkl')
    main()
