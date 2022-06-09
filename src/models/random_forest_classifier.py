import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.features.helpers import absolute_path
from src.data.make_dataset import load_labels, load_train_data


def get_random_forest_classifier(X_train, y_train, cache=True):
    if os.path.exists(RFC_DUMP_FILE) and cache:
        clf = joblib.load(open(RFC_DUMP_FILE, 'rb'))
    else:
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        joblib.dump(clf, RFC_DUMP_FILE)

    return clf



def score_random_forest(X_train, y_train, X_test, y_test):
    clf = get_random_forest_classifier(X_train, y_train, False)
    y_pred = clf.predict(X_test)
    print("RandomForestClassifier Accuracy:", accuracy_score(y_test, y_pred))


def main():
    X = load_train_data()
    y = load_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2022)
    score_random_forest(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    RFC_DUMP_FILE = absolute_path('models', 'random_forest_classifier.pkl')
    main()
