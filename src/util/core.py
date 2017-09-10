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

def sigmoid_forward(z):
	cache = {
		"Z" : z
	} 
	return 1 / (1 + np.exp(-z)), cache

def relu_forward(z):
	cache = {
		"Z" : z
	}
	if z > 0:
		return z, cache
	return 0, cache

def sigmoid_backward(dA, cache):
	Z = cache["Z"]
	dAdZ = Z * (1 - Z)
	dZ = np.multiply(dZ, dAdZ)	
	return dZ

def relu_backward(dA, cache):
	Z = cache["Z"]
	if Z > 0:
		return dA
	return 0

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


def initialize_parameters_deep(layer_dims):
	parameters = {} 
	L = len(layer_dims)
	for l in range(1, L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

	return parameters

def linear_forward(A_prev, W, b):
	assert(W.shape[1] == A_prev.shape[0])
	assert(W.shape[0] == b.shape[0])

	Z = np.dot(W, A_prev) + b
	cache = (A_prev, W, b)

	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	Z, linear_cache = linear_forward(A_prev, W, b)

	if activation == "sigmoid":
		A, activation_cache = sigmoid_forward(Z)
	elif activation == "relu":
		A, activation_cache = relu_forward(Z)

	assert(A.shape == (W.shape[0], A_prev.shape[1]))
	cache = linear_cache, activation_cache
	
	return A, cache

def L_model_forward(X, parameters):

	A = X
	caches = {}
	#- Number of layers in neural network
	L = len(parameters) // 2

	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
	
	caches.append(cache)
	assert(AL.shape == (1, X.shape[1]))

	return AL, caches

def linear_backward(dZ, cache):
	A_prev, W, b = cache

	m = A_prev.shape[1]
	
	dW = (1 / m) * np.dot(dZ, A_prev.T)
	db = (1 / m) * np.sum(dZ, axis=1)
	dA_prev = np.dot(W.T, dZ)

	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape ==b.shape)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	
	if activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
	elif activation == "relu":
		dZ = relu_backward(dA, activation_cache)
	
	dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
	grad = {}
	#- Number of layers in the models
	L = len(caches)	
	m = AL.shape[1]
	Y = np.reshape(AL.shape)

	#- dJ/dAL
	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	#- Backward prop for layer L
	current_cache = caches[L - 1]
	grad["dA" + str(L)], grad["dW" + str(L)], grad["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

	for l in  reversed(range(0, L - 1)):
		current_cache = caches[l]
		dA_prev, dW, db = linear_activation_backward(grad["dA" + str(l + 2)], current_cache, activation = "relu")
		grad["dA" + str(l + 1)] = dA_prev
		grad["dW" + str(l + 1)] = dW
		grad["db" + str(l + 1)] = db
	
	return grads

def update_parameters(parameters, grads, learning_rate):
	#- Number of layers in the models
	L = len(parameters) // 2
	for l in range(0, L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

	return parameters
