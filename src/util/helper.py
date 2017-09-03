import csv
import numpy as np

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
		elif train_embarked[0][i] == "C":
			test_embarked[0][i] = 2
		else:
			test_embarked[0][i] = 3
	
	train_dict["Embarked"] = train_embarked
	test_dict["Embarked"] = test_embarked

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