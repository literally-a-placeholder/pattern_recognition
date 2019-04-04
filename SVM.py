# Use Python 3.7
# Group literally_a_placeholder
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():
    train = FileLoader('./datasets/mnist_train.csv')  # TODO replace with real files
    test = FileLoader('./datasets/mnist_test.csv') # TODO replace with real files
    n_folds = 3
    classifier = parameter_tuning(train, n_folds=n_folds)
    validation_score = round(validate(classifier, test), 3)*100
    print_results(classifier, validation_score)


def parameter_tuning(train, n_folds):
    print('Started parameter tuning...')
    Cs = [0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1]
    params = [
        {'kernel': ['linear'], 'C': Cs},
        {'kernel': ['rbf'], 'C': Cs, 'gamma': gammas},
    ]
    classifier = GridSearchCV(estimator=svm.SVC(), param_grid=params, n_jobs=-1, cv=n_folds)

    print('Fitting to training set...')
    classifier.fit(train.data, train.target)
    return classifier


def validate(classifier, test):
    print('Validating classifier on test set...')
    return classifier.score(test.data, test.target)


def print_results(classifier, validation_score):
    print('Cross Validation Results:')
    print('\tAverage accuracy during cross-validation: ', round(classifier.best_score_, 3)*100, '%')
    print('\tBest Kernel:', classifier.best_estimator_.kernel)
    print('\tBest C:', classifier.best_estimator_.C)
    if classifier.best_estimator_.kernel == 'rbf':
        print('\tBest Gamma:', classifier.best_estimator_.gamma)

    print('\nTotal accuracy on test set: ', validation_score, '%')


class FileLoader:
    def __init__(self, filename):
        csvfile = self.load(filename)
        cols = len(csvfile[0])
        self.target = csvfile[:, 0]
        self.data = csvfile[:, 1:cols]

    @staticmethod
    def load(filename):
        print('Loading ' + filename + '...')
        csvfile = np.loadtxt(filename, delimiter=',', dtype=int)
        return csvfile


if __name__ == '__main__':
    main()


# inspired by the following tutorials:
# https://chrisalbon.com/machine_learning/model_evaluation/cross_validation_parameter_tuning_grid_search/
# https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
