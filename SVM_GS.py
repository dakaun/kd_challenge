import IOHandler
import PreprocessingHandler

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

data = IOHandler.read_data()
num_data = PreprocessingHandler.extract_numerical(data)
X, y = PreprocessingHandler.get_X_y(num_data)

X_train, X_test, y_train, y_test = train_test_split(X, y)

svc = SVC()
svc.fit(X_train, y_train)

params = {'kernel': ('poly', 'sigmoid'),
          'C': [1,10]}
gridsearch = GridSearchCV(svc, params, cv=5)
gridsearch.fit(X_train, y_train.values.ravel())
print(gridsearch)
y_pred_gs = gridsearch.predict(X_test)
gs_score = gridsearch.score(X_test, y_pred_gs)

#svc_score = svc.score(X_test, y_test)