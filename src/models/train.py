from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def verify_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    print("Cross val score:", np.mean(cross_val_score(clf, X_test, y_test, cv=kfold, scoring='balanced_accuracy')))
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
