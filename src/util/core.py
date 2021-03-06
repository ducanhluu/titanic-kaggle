import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')

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
	W1 = np.random.randn(n_y, n_x) * 0.01
	b1 = np.zeros((n_y, 1))

	parameters = {
		"W" : W1,
		"b" : b1
	}

	return parameters

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoid_forward(z):
	g = 1 / (1 + np.exp(-z))
	cache = {
		"G" : g
	}
	
	return g, cache

def relu_forward(Z):
	cache = {
		"Z" : Z
	}
	G = Z
	G[Z <= 0] = 0
	
	return G, cache

def sigmoid_backward(dA, cache):
	G = cache["G"]
	dAdZ = G * (1 - G)
	dZ = np.multiply(dA, dAdZ)

	return dZ

def relu_backward(dA, cache):
	Z = cache["Z"]
	dZ = dA
	dZ[Z <= 0] = 0

	return dZ

def compute_cost(Y_pred, Y):
	m = Y.shape[1]
	cost = (-1 / m) * np.sum(np.multiply(Y, np.log(Y_pred)) + np.multiply((1 - Y), np.log(1 - Y_pred)), axis=1)
	cost = np.squeeze(cost)
	return cost

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

		if i % 100 == 0 and print_cost:
			print("Cost after iteration %i: %f" %(i, cost))
			costs.append(cost)

	params = {
		"W" : W,
		"b" : b
	}

	return params, costs

def logistic_regression_predict(X, parameters):
	W = parameters["W"]
	b = parameters["b"]
	Y_pred = sigmoid(np.dot(W, X) + b) > 0.5
	return Y_pred

def model(X_train, Y_train, num_epochs=2000, learning_rate = 0.5, print_cost = False):
	parameters = initialize_parameters(X_train.shape[0], Y_train.shape[0])
	W = parameters["W"]
	b = parameters["b"]

	params, costs = optimize(W, b, X_train, Y_train, num_epochs, learning_rate, print_cost)

	plt.plot(np.squeeze(costs))
	plt.ylabel("cost")
	plt.xlabel("iterations (per hundreds)")
	plt.title("Logistics regression with learning rate = " + str(learning_rate))
	plt.show()

	return params


def initialize_parameters_deep(layers_dims):
	parameters = {} 
	L = len(layers_dims)
	for l in range(1, L):
		parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
		parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

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
	caches = []
	#- Number of layers in neural network
	L = len(parameters) // 2
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
	
	caches.append(cache)
	assert(AL.shape == (1, X.shape[1]))

	return AL, caches

def linear_backward(dZ, cache):
	A_prev, W, b = cache

	m = A_prev.shape[1]
	
	dW = (1 / m) * np.dot(dZ, A_prev.T)
	db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
	dA_prev = np.dot(W.T, dZ)

	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)

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
	grads = {}
	#- Number of layers in the models
	L = len(caches)	
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	#- dJ/dAL
	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	#- Backward prop for layer L
	current_cache = caches[L - 1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

	for l in  reversed(range(0, L - 1)):
		current_cache = caches[l]
		dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
		grads["dA" + str(l + 1)] = dA_prev
		grads["dW" + str(l + 1)] = dW
		grads["db" + str(l + 1)] = db
	
	return grads

def update_parameters(parameters, grads, learning_rate):
	#- Number of layers in the models
	L = len(parameters) // 2
	for l in range(0, L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

	return parameters


def L_layers_model(X, Y, layers_dims, num_epochs=2000, learning_rate = 0.5, print_cost = False):
	costs = []
	#- Initialization of parameters
	parameters = initialize_parameters_deep(layers_dims)

	#- Traning process
	for i in range(num_epochs):
		#-- Forward propagation
		AL, caches = L_model_forward(X, parameters)
		
		#-- Computing cost to keep trace
		cost = compute_cost(AL, Y)

		#-- Backward propagation
		grads = L_model_backward(AL, Y, caches)

		#-- Updating parameters after an iteration
		parameters = update_parameters(parameters, grads, learning_rate)

		#-- Printing cost to keep trace
		if print_cost and i % 100 == 0:
			print("Cost after iteration %i: %f" %(i, cost))
			costs.append(cost)

	plt.plot(np.squeeze(costs))
	plt.ylabel("cost")
	plt.xlabel("iterations (per hundreds)")
	plt.title("Deep learning with learning rate = " + str(learning_rate))
	plt.show()

	return parameters

def L_layers_predict(X, parameters):
	AL, _ = L_model_forward(X, parameters)
	Y_pred = AL > 0.5
	return Y_pred