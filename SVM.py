# Use Python 3.7
# Group literally_a_placeholder
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():
    train = FileLoader('./datasets/mnist_train_dev.csv')  # TODO replace with real files
    test = FileLoader('./datasets/mnist_test_dev.csv') # TODO replace with real files
    n_folds = 10
    default_perf = run_std_svm(train, test)
    classifier = parameter_tuning(train, n_folds=n_folds)
    validation_score = round(validate(classifier, test), 3)*100
    print_results(default_perf, classifier, validation_score)


def run_std_svm(train, test):
    print('Running standard SVM before parameter tuning...')
    linear_SVM = svm.SVC(kernel='linear')
    rbf_SVM = svm.SVC()  # default is rbf

    linear_SVM.fit(train.data, train.target)
    linear_test_score = round(linear_SVM.score(test.data, test.target), 3)*100

    rbf_SVM.fit(train.data, train.target)
    rbf_test_score = round(rbf_SVM.score(test.data, test.target), 3)*100

    return (linear_test_score, rbf_test_score)


def parameter_tuning(train, n_folds):
    print('Started parameter tuning...')
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
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


def print_results(default_perf, classifier, validation_score):
    print('Accuracy before parameter tuning for: ')
    print('\tlinear kernel: ', default_perf[0], '%')
    print('\trbf kernel: ', default_perf[1], '%\n')

    print('Cross Validation Results:')
    print('\tAverage accuracy during cross-validation: ', round(classifier.best_score_, 3)*100, '%')
    print('\tBest Kernel:', classifier.best_estimator_.kernel)
    print('\tBest C:', classifier.best_estimator_.C)
    if classifier.best_estimator_.kernel == 'rbf':
        print('\tBest Gamma:', classifier.best_estimator_.gamma)

    print('\nTotal accuracy on test set: ', validation_score, '%')
    print('\tAccuracy increase using parameter tuning:', validation_score - max(default_perf), '%')


def whatever(accuracy, classifier):
    print(' Accuracy on Test Set: ', round(accuracy, 4) * 100, '% for:')
    print('Best Kernel:', classifier.best_estimator_.kernel)
    print('Best C:', classifier.best_estimator_.C)
    if classifier.best_estimator_.kernel == 'rbf':
        print('Best Gamma:', classifier.best_estimator_.gamma)
    print(classifier.cv_results_)


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
