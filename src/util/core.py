import numpy as np

def initialize_parameters(n_x, n_y):
	"""
    Argument:
    n_x -- size of the input layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_y, n_x)
                    b1 -- bias vector of shape (n_y, 1)
    """

	W1 = np.zeros((n_y, n_x))
	b1 = np.zeros((n_y, 1))

	parameters = {
		"W" : W1,
		"b" : b1
	}

	return parameters

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def compute_cost(Y_pred, Y):
	m = Y.shape[1]
	return (-1 / m) * np.sum(np.multiply(Y, np.log(Y_pred)) + np.multiply((1 - Y), np.log(1 - Y_pred)), axis=1)

def propagate(W, b, X, Y):
	m = X.shape[1]
	Z = np.dot(W, X) + b
	A = sigmoid(Z)
	cost = np.squeeze(compute_cost(A, Y))

	dW = (1 / m) * np.dot(A - Y, X.T)
	db = (1 / m) * np.sum(A - Y, axis=1)

	grads = {
		"dW" : dW,
		"db" : db
	}

	return grads, cost

def optimize(W, b, X, Y, num_epochs, learning_rate, print_cost = False):
	costs = []
	for i in range(num_epochs):
		grads, cost = propagate(W, b, X, Y)
		dW = grads["dW"]
		db = grads["db"]

		W = W - learning_rate * dW
		b = b - learning_rate * db

		if i % 100 == 0:
			costs.append(cost)

		if i % 100 == 0 and print_cost:
			print("Cost after iteration %i: %f" %(i, cost))

	params = {
		"W" : W,
		"b" : b
	}

	return params, costs

def predict(W, b, X):
	m = X.shape[1]
	Y_pred = sigmoid(np.dot(W, X) + b) > 0.5
	return Y_pred

def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, num_epochs=2000, learning_rate = 0.5, print_cost = False):
	parameters = initialize_parameters(X_train.shape[0], Y_train.shape[0])
	W = parameters["W"]
	b = parameters["b"]

	params, costs = optimize(W, b, X_train, Y_train, num_epochs, learning_rate, print_cost)

	W = params["W"]
	b = params["b"]

	Y_train_pred = predict(W, b, X_train)
	Y_dev_pred = predict(W, b, X_dev)
	Y_test_pred = predict(W, b, X_test)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_pred - Y_train)) * 100))
	print("dev accuracy: {} %".format(100 - np.mean(np.abs(Y_dev_pred - Y_dev)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_pred - Y_test)) * 100))
