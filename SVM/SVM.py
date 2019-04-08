# Use Python 3.7
# Group literally_a_placeholder
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():
    train = FileLoader('./datasets/mnist_train_dev.csv')  # TODO replace with real files
    test = FileLoader('./datasets/mnist_test_dev.csv') # TODO replace with real files
    n_folds = 3
    classifier = parameter_tuning(train, n_folds=n_folds)
    validation_score = round(validate(classifier, test), 3)*100
    print_results(classifier, validation_score)


def parameter_tuning(train, n_folds):
    params = [
        {'C': [0.1, 1, 10], 'kernel': ['linear'], },
        {'C': [10, 1, 0.1], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf'], },
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

    print('\tGrid Scores on Training:')
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("\t\t%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('\tBest Params:', classifier.best_params_)

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
