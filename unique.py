import numpy as np


def Unique(arr):
	# Compute the number of unique elements in an integer array.
	print("searching for unique elements in\n{}".format(arr))
	size = arr.shape[0]
	unique_indicator = np.ones(size, dtype = np.int)
	for i in range(size):
		for j in range(i + 1, size):
			if (arr[i] == arr[j]):
				unique_indicator[j] = 0
	
	# Count the unique elements
	n_unique = np.sum(unique_indicator)

	print("{} unique elements\n{}".format(n_unique, unique_indicator))
	
	# Load the unique elements in the array
	unique = np.zeros(n_unique, dtype = np.int)
	n_unique = 0
	for i in range(size):
		if (unique_indicator[i] == 1):
			unique[n_unique] = i
			n_unique = n_unique + 1
	return n_unique


def SupportToLabel(interactions):
	# Convert a list of qubit indices to labels for a tensor.
	# Each qubit index corresponds to a pair of labels, indicating the row and column indices of the 2 x 2 matrix which acts non-trivially on that qubit.
	# Each number in the support is mapped to a pair of alphabets in the characters list, as: x -> (characters[2x], characters[2x + 1]).
	# Eg. (x, y, z) ---> (C[2x] C[2y] C[2z] , C[2x + 1] C[2y + 1] C[2z + 1])
	n_interactions = interactions.shape[0]
	max_interaction_range = np.max(interactions[:, 0])
	contraction_labels = np.zeros((n_interactions, max_interaction_range, 2), dtype = np.int)
	
	n_symbols = 0
	for q in range(n_interactions):
		n_symbols += 2 * interactions[q, 0]
	symbols = np.arange(n_symbols, dtype = np.int)
	
	# Compute the unique qubits in the list of interactions.
	qubits = np.zeros(n_symbols//2, dtype = np.int)
	n_qubits = 0
	for i in range(n_interactions):
		for q in range(1, 1 + interactions[i, 0]):
			qubits[n_qubits] = interactions[i, q]
			n_qubits = n_qubits + 1
	
	unique_qubits = np.unique(qubits)
	n_unique = unique_qubits.shape[0]

	print("{} unique qubits\n{}".format(n_unique, unique_qubits))
	
	# Free indices
	free_index = np.zeros((n_unique, 2), dtype = np.int)
	for q in range(n_unique):
		free_index[q, 0] = -1
		free_index[q, 1] = -1
	
	# Initialize contraction labels
	for i in range(n_interactions):
		for q in range(interactions[i, 0]):
			contraction_labels[i, q, 0] = -1
			contraction_labels[i, q, 1] = -1

	# Compute the contraction labels and free indices according to the interactions.
	for i in range(n_interactions):
		for j in range(interactions[i, 0]):
			q = interactions[i, j + 1]

			if (free_index[q, 0] == -1):
				n_symbols -= 1
				free_index[q, 0] = symbols[n_symbols]
				n_symbols -= 1
				free_index[q, 1] = symbols[n_symbols]

				contraction_labels[i, j, 0] = free_index[q, 0]
				contraction_labels[i, j, 1] = free_index[q, 1]

			else:
				contraction_labels[i, j, 0] = free_index[q, 1]
				n_symbols -= 1
				free_index[q, 1] = symbols[n_symbols]
				contraction_labels[i, j, 1] = free_index[q, 1]

	return (contraction_labels, free_index, unique_qubits)