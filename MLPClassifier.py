from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph
from sklearn.model_selection import cross_val_score
import numpy as np


def mlp_classifier(X_train,X_test,y_train,y_test, X_total, y, x_val=False):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,1), random_state=1)
    if x_val:
        v_score = cross_val_score(clf, X_total, np.ravel(y), scoring='accuracy', cv=5)
        return print(f'MLP Classifier Score is : {str(round(100*v_score.mean(), 2))}')
    else:
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        dt_score = clf.score(X_test, y_test)
        return print(f'MLP Classifier Score is : {dt_score}')
