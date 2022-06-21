import sys
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import time
sys.path.append('C:\\dsproject\\ml-project')
import joblib
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from src.data.make_dataset import load_labels, load_train_data, load_raw_train_data
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator, any_classifier, svc
from hyperopt import tpe
import datetime
from src.features.helpers import absolute_path
import logging


def _train_test_data(X, y, args, test_size=0.3, random_state=542):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    if args.scaler:
        logging.info("Before StandardScaler")
        sc = StandardScaler(with_std=False)
        X_train = sc.fit_transform(X_train,y_train)
        X_test = sc.transform(X_test)

    if args.smote:
        logging.info("Before Smote")
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if args.pca:
        logging.info("Before Pca. n_components:", str(args.n_components))
        pca = PCA(n_components=args.n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test


def _check_condition(args, itter):
    if args.hour > 0:
        return (time.time() - starttime) / 3600 < args.hour

    return itter < args.try_number


def _get_args():
    parser = argparse.ArgumentParser("Hyperopt Search")
    parser.add_argument("--evals", help="Hyperopt max_evals param - Default 10", type=int, default=10)
    parser.add_argument("--scaler", help="Apply StandardScaler - Default True", type=bool, default=True)
    parser.add_argument("--pca", help="Apply PCA - Default True", type=bool, default=True)
    parser.add_argument("--n_components", help="PCA n_components - Default 0.98", type=float, default=0.98)
    parser.add_argument("--smote", help="Apply smote - Default True", type=bool, default=True)
    parser.add_argument("--try_number", help="Number of executions HyperoptEstimator.fit() in loop - Default 1",
                        type=int,
                        default=1)
    parser.add_argument("--hour", help="Minimum script execution time in hour - Default disable", type=int,
                        default=0)
    parser.add_argument("--trial_timeout", help="HyperoptEstimator trial_timeout param - Default 90",
                        type=int, default=90)
    parser.add_argument("--log_level", help="Log level - Default 50", type=int, default=10)
    parser.add_argument("--min_score", help="Min roc auc score to save model", type=float, default=90)
    return parser.parse_args()


if __name__ == "__main__":
    starttime = time.time()
    args = _get_args()
    logging.basicConfig(encoding='utf-8', level=args.log_level)

    X = load_raw_train_data()
    y = load_labels()

    X_train, X_test, y_train, y_test = _train_test_data(X, y, args, test_size=0.3, random_state=542)

    itter = 0
    while _check_condition(args, itter, starttime):
        itter = itter + 1
        try:
            hyperEstim = HyperoptEstimator(classifier=svc('my_svc'), max_evals=args.evals, algo=tpe.suggest,
                                           n_jobs=-1, trial_timeout=args.trial_timeout)
            hyperEstim.fit(X_train, y_train.values.ravel())
            y_pred = hyperEstim.predict(X_test)
            logging.info("Best ROC: ", roc_auc_score(y_test, y_pred))
            logging.info(hyperEstim.best_model())
            if roc_auc_score(y_test, y_pred) > args.min_score:
                dump_file = absolute_path('models', 'hyperopt_estimator_' + str(itter) + '.pkl')
                logging.info("Saveing model. ROC:", roc_auc_score(y_test, y_pred))
                joblib.dump(hyperEstim, dump_file)
        except Exception as ex:
            logging.error(ex.__str__())
    print(itter)







