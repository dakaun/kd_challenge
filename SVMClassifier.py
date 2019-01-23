from sklearn import svm
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph
from sklearn.model_selection import cross_val_score
import numpy as np

def svm_classifier(X_train,X_test,y_train,y_test, X_total, y, x_val=False):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = svm.SVC(gamma='scale') #(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False),
    if x_val:
        v_score = cross_val_score(clf, X_total, np.ravel(y),scoring='accuracy', cv=5)
        return print(f'SVM Classifier Score is : {str(round(100*v_score.mean(), 2))}')
    else:
        clf.fit(X_train, np.ravel(y_train))
        y_pre = clf.predict(X_test)
        dt_score = clf.score(X_test, y_test)
        return print(f'SVM Classifier Score is : {dt_score}')