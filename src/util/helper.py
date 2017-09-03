import csv
import numpy as np
from sklearn.utils import shuffle

def load_csvfile(filename):
	data_dict = {}
	with open(filename, "r") as csvfile:
		reader = csv.reader(csvfile)
		header = next(reader)
		for h in header:
			data_dict[h] = []
		for row in reader:
			for h, v in zip(header, row):
				data_dict[h].append(v)
		for h in header:
			data_dict[h] = np.reshape(data_dict[h], (1, -1))
	return data_dict, header

def handle_data(train_file, test_file):
	train_dict, train_header = load_csvfile(train_file)
	test_dict, _ = load_csvfile(test_file)

	train_set_y = train_dict["Survived"]
	train_age = train_dict["Age"]
	train_age[train_age == ''] = 0
	train_dict["Age"] = train_age
	
	test_age = test_dict["Age"]
	test_age[test_age == ''] = 0
	test_dict["Age"] = test_age
	

	train_dict["Sex"] = (train_dict["Sex"] == "male").astype(int)
	test_dict["Sex"] = (test_dict["Sex"] == "male").astype(int)
	
	train_embarked = train_dict["Embarked"]
	test_embarked = test_dict["Embarked"]
	for i in range(0, train_embarked.shape[1]):
		if train_embarked[0][i] == "S":
			train_embarked[0][i] = 1
		elif train_embarked[0][i] == "C":
			train_embarked[0][i] = 2
		else:
			train_embarked[0][i] = 3

	for i in range(0, test_embarked.shape[1]):
		if test_embarked[0][i] == "S":
			test_embarked[0][i] = 1
		elif test_embarked[0][i] == "C":
			test_embarked[0][i] = 2
		else:
			test_embarked[0][i] = 3
	
	train_dict["Embarked"] = train_embarked
	test_dict["Embarked"] = test_embarked
	
	fare_test = test_dict["Fare"]
	fare_test[fare_test == ''] = 0
	test_dict["Fare"] = fare_test

	train_header.remove("PassengerId")
	train_header.remove("Survived")
	train_header.remove("Cabin")
	train_header.remove("Pclass")
	train_header.remove("Name")
	train_header.remove("Ticket")
	train_set_x = np.array(train_dict["Pclass"])
	test_set_x = np.array(test_dict["Pclass"])

	for h in train_header:
		train_set_x = np.concatenate((train_set_x, train_dict[h]))
		test_set_x = np.concatenate((test_set_x, test_dict[h]))
	
	return train_set_x, train_set_y, test_set_x

def load_dataset(train_file, test_file):
	return	handle_data(train_file, test_file)

def shuffle_data(X, Y):
	order = np.arange(0, X.shape[1]).reshape((1, -1))
	X, Y, order =  shuffle(X.T, Y.T, order.T, random_state=0)
	return X.T, Y.T, order.T

def normalize(X):
	m = X.shape[1]
	means = (1 / m) * np.sum(X, axis=1, keepdims=True)
	X = X - means
	var = (1 / m) * np.sum(np.multiply(X, X), axis=1, keepdims=True)
	X = X / np.sqrt(var)
	return X

def divide_dataset(size, X, Y):
	# @param size	the percent of dataset into dev set

	m = X.shape[1]
	dev_size = np.floor(m * size / 100).astype(int)
	X_train = normalize(np.array(X[:, 0:(m - 2 * dev_size)], dtype=np.float64))
	Y_train = np.array(Y[:, 0:(m - 2 * dev_size)], dtype=np.float64)

	X_dev = normalize(np.array(X[:, (m - 2 * dev_size):(m - dev_size)], dtype=np.float64))
	Y_dev = np.array(Y[:, (m - 2 * dev_size):(m - dev_size)], dtype=np.float64)

	X_test = normalize(np.array(X[:, (m - dev_size):], dtype=np.float64))
	Y_test = np.array(Y[:, (m - dev_size):], dtype=np.float64)
	return X_train, Y_train, X_dev, Y_dev, X_test, Y_test