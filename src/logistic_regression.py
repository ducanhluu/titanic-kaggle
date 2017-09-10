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
#- Logistic regression
parameters = model(X_train, Y_train, num_epochs = 2000, learning_rate = 0.5, print_cost = True)

Y_train_pred = logistic_regression_predict(X_train, parameters)
Y_dev_pred = logistic_regression_predict(X_dev, parameters)
Y_test_pred = logistic_regression_predict(X_test, parameters)

#- Deep neural networks
# layers_dims = [X_train.shape[0], Y_train.shape[0]]
# parameters = L_layers_model(X_train, Y_train, layers_dims, num_epochs = 2000, learning_rate = 0.5, print_cost = True)

# Y_train_pred = L_layers_predict(X_train, parameters)
# Y_dev_pred = L_layers_predict(X_dev, parameters)
# Y_test_pred = L_layers_predict(X_test, parameters)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_pred - Y_train)) * 100))
print("dev accuracy: {} %".format(100 - np.mean(np.abs(Y_dev_pred - Y_dev)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_pred - Y_test)) * 100))