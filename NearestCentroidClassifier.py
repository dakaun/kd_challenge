from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph

def nc_classifier(X_train,X_test,y_train,y_test):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = clf = NearestCentroid().fit(X_train, y_train)

    y_pre = clf.predict(X_test)
    dt_score = clf.score(X_test, y_test)
    print(f'Nearest Centroid Classifier Score is : {dt_score}')