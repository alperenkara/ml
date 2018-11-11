# Simple linear regression on the Swedish Insurance Dataset

from random import seed
from random import randrange
from csv import reader
from math import sqrt

# load the dataset

def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if row == NULL:
                continue
            dataset.append(row)
    return dataset

# tranform string to float 
def str_column_to_float(dataset, column):
    for row in dataset:
        # whitespaces are removed
        row[column] = float(row[column].strip())

# split a dataset into a train and test set

def train_test_split(dataset, split):
    train = list()
    train_size = split*len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy