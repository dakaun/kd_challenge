from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import PreprocessingHandler as ph


def mlp_classifier(data):
    X, y = ph.get_X_y(data)
    # X_stand = ph.standard(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,1), random_state=1).fit(X_train, y_train)

    #Todo: Vary the number of hidden layers => empirical approach

    y_pre = clf.predict(X_test)
    dt_score = clf.score(X_test, y_test)
    print(f'MLP Classifier Score is : {dt_score}')
