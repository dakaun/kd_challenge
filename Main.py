# Main file for running the analysis

import IOHandler as io
import PreprocessingHandler as ph
import SVMClassifier as svm
import pandas as pd


def main_testing():
    # ToDo describe methods for determining score of SVM


def main_predict():
    training_data = io.read_data('train')
    test_data = io.read_data('test')

    training_data_numerical = ph.extract_numerical(training_data)
    test_data_numerical = ph.extract_numerical(test_data)

    scaler = ph.define_standard_scaler(training_data_numerical)
    training_data_numerical_st = ph.fit_standard_scaler(training_data_numerical, scaler)
    test_data_numerical_st = ph.fit_standard_scaler(test_data_numerical, scaler)

    training_data_label = training_data[['label']]
    svm_clf = svm.train_svm_classifier(training_data_numerical_st, training_data_label)

    prediction = svm.predict_test_data(svm_clf, test_data_numerical_st)

    # print(prediction)
    test_labels = pd.DataFrame(data=prediction, columns=['label'])

    io.append_results_to_csv(test_labels)

main_predict()