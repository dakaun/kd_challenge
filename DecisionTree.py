from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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


def decision_tree(data):
    X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    decisiontree = DecisionTreeClassifier()
    decisiontree.fit(X_train, y_train)
    y_pre = decisiontree.predict(X_test)
    dt_score = decisiontree.score(X_test, y_test)
    print(f'Decision tree Score is : {dt_score}')
