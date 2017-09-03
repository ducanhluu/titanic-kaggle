from util import *
import numpy as np

train_set = "./data/train.csv"
test_set = "./data/test.csv"
#- Load train and set data
train_set_x_orig, train_set_y, test_set_x_orig = load_dataset(train_set, test_set)

#- Shuffle data
train_set_x_orig, train_set_y, order = shuffle_data(train_set_x_orig, train_set_y)

#- Devide dataset into train/dev/test set
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = divide_dataset(20, train_set_x_orig, train_set_y)

model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, num_epochs=20000, learning_rate = 0.05, print_cost = True)