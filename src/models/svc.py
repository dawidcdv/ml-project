import argparse
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from src.data.dataset import load_labels, load_raw_train_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.features.helpers import absolute_path
from src.models.train_helper import verify_model


def _get_args():
    parser = argparse.ArgumentParser("SVC")
    parser.add_argument("--resample", help="Apply Smote and RandomUderSampler - Default False", type=bool,
                        default=False)
    return parser.parse_args()


def create_svc_classifier():
    return SVC(C=1.0752079389673375, coef0=0.043809020064986326, degree=2, gamma='auto',
        kernel='poly', random_state=4, shrinking=False, tol=9.287881080560548e-05)


def main():
    args = _get_args()
    X = load_raw_train_data()
    y = load_labels()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    sc = StandardScaler(with_std=False)
    X_train = sc.fit_transform(X_train,y_train)
    X_test = sc.transform(X_test)

    if args.resample:
        smote = SMOTE()
        under = RandomUnderSampler()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_train, y_train = under.fit_resample(X_train, y_train)


    classifier_svc = create_svc_classifier()
    verify_model(classifier_svc, X_train, y_train, X_test, y_test, absolute_path("reports","figures","svc_cm.jpg"))


if __name__ == '__main__':
    main()
