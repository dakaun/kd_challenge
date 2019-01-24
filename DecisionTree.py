from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

import PreprocessingHandler as ph


# data = IOHandler.read_data('num_genre')
# data = pd.read_csv('./data/num_genre.csv')
# X, y = PreprocessingHandler.get_X_y(data)
# X_stand = PreprocessingHandler.standard(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# decisiontree = DecisionTreeClassifier()
# decisiontree.fit(X_train, y_train)
# y_pre = decisiontree.predict(X_test)
# dt_score = decisiontree.score(X_test, y_test)
# print(f'Score is : {dt_score}')


def decision_tree(X_train,X_test,y_train,y_test, X_total, y, x_val=False):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    decisiontree = DecisionTreeClassifier()

    if x_val:
        v_score = cross_val_score(decisiontree, X_total, np.ravel(y),scoring='accuracy', cv=10)
        return print(f'Decision tree Score is : {str(round(100*v_score.mean(), 2))}')
    else:
        decisiontree.fit(X_train, y_train)
        y_pre = decisiontree.predict(X_test)
        dt_score = decisiontree.score(X_test, y_test)
        return print(f'Decision tree Score is : {dt_score}')
