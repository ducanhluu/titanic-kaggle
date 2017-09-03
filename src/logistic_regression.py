from util import *
import numpy as np

train_set = "./data/train.csv"
test_set = "./data/test.csv"
#- Load train and set data
train_set_x_orig, train_set_y, test_set_x_orig = load_dataset(train_set, test_set)

#- Shuffle data
train_set_x_orig, train_set_y, order = shuffle_data(train_set_x_orig, train_set_y)

#- Devide dataset into train/dev/test set

x_train, y_train, x_dev, y_dev, x_test, y_test = divide_dataset(20, train_set_x_orig, train_set_y)
print(x_train[:, 0])