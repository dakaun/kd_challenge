from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np


def gnb_classifier(X_train,X_test,y_train,y_test, X_total, y, x_val=False):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = GaussianNB()

    if x_val:
        v_score = cross_val_score(clf, X_total, np.ravel(y), scoring='accuracy', cv=5)#fit_params= (parameters für gaussiannb
        return print(f'GNB Classifier Score is : {str(round(100*v_score.mean(), 2))}')
    else:
        clf.fit(X_train, np.ravel(y_train))
        y_pre = clf.predict(X_test)
        dt_score = clf.score(X_test, y_test)
        return print(f'GNB Classifier Score is : {dt_score}')