from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np

def nc_classifier(X_train,X_test,y_train,y_test, X_total, y, x_val=False):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = clf = NearestCentroid()

    if x_val:
        v_score = cross_val_score(clf, X_total, np.ravel(y),scoring='accuracy', cv=ShuffleSplit(n_splits=10, test_size=.25, train_size=None))
        return print(f'Nearest Centroid Classifier Score is : {str(round(100*v_score.mean(), 2))}')
    else:
        clf.fit(X_train, np.ravel(y_train))
        y_pre = clf.predict(X_test)
        dt_score = clf.score(X_test, y_test)
        return print(f'Nearest Centroid Classifier Score is : {dt_score}')