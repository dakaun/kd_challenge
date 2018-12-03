import IOHandler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# extract numerical columns
def extract_numerical(dataset):
    subset = dataset[['label', 'budget', 'popularity', 'revenue', 'runtime']]
    return subset

def drop_columns(dataset, column_list):
    for column in column_list:
        dataset = dataset.drop(column, axis=1)
    return dataset

def get_X_y(dataset):
    target = dataset.iloc[:, :1]
    X = dataset.iloc[:, 1:]
    return X, target



data = IOHandler.read_data()
data = extract_numerical(data)
#data = drop_columns(data, ['imdbId', 'tmdbId'])
X, y = get_X_y(data)
print(type(X))
print(X)
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y)

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pre = decisiontree.predict(X_test)
dt_score = decisiontree.score(X_test, y_test)
print(f'Score is : {dt_score}')