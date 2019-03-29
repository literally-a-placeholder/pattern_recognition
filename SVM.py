# Use Python 3.7

import numpy as np


def main():
    train = FileLoader('datasets/train.csv')
    data = train.data
    test = train.test


class FileLoader:
    def __init__(self, filename):
        csvfile = self.load(filename)
        cols = len(csvfile[0])
        self.target = csvfile[:, 0]
        self.data = csvfile[:, 1:cols]

    @staticmethod
    def load(filename):
        print('Loading data...')
        csvfile = np.loadtxt(filename, delimiter=',', dtype=int)
        return csvfile


if __name__ == '__main__':
    main()
