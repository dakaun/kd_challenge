from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph

def gnb_classifier(X_train,X_test,y_train,y_test):
    #X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = GaussianNB().fit(X_train, y_train)

    y_pre = clf.predict(X_test)
    dt_score = clf.score(X_test, y_test)
    print(f'GNB Classifier Score is : {dt_score}')