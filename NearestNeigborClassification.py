from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph
from sklearn.model_selection import cross_val_score
import numpy as np

def knn_classifier(X_train,X_test,y_train,y_test, X_total, y, x_val=False):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = KNeighborsClassifier(n_neighbors=5, p=2) # (cv=5, error_score='raise-deprecating', estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=5, p=2,weights='uniform')
    if x_val:
        v_score = cross_val_score(clf, X_total, np.ravel(y),scoring='accuracy', cv=5)
        return print(f'KNN Classifier Score is : {str(round(100*v_score.mean(), 2))}')
    else:
        # Todo: Vary the number of n_neighbors => empirical approach
        clf.fit(X_train, np.ravel(y_train))
        y_pre = clf.predict(X_test)
        dt_score = clf.score(X_test, y_test)
        return print(f'KNN Classifier Score is : {dt_score}')
