import numpy as np


def Unique(arr, size, unique, return_elements):
	# Compute the number of unique elements in an integer array.
	
	print("searching for unique elements in\n{}".format(arr))
	
	unique_indicator = np.ones(size, dtype = np.int32)
	for i in range(size):
		for j in range(i + 1, size):
			if (arr[i] == arr[j]):
				unique_indicator[j] = 0
	
	# Count the unique elements
	n_unique = np.sum(unique_indicator)

	print("%d unique elements" % (n_unique))
	print("unique_indicator\n{}".format(unique_indicator))
	
	if (return_elements == 0):
		# Load the unique elements in the array
		n_unique = 0
		for i in range(size):
			if (unique_indicator[i] == 1):
				unique[n_unique] = i
				n_unique = n_unique + 1
	return n_unique


def SupportToLabel(interactions, contraction_labels):
	# Convert a list of qubit indices to labels for a tensor.
	# Each qubit index corresponds to a pair of labels, indicating the row and column indices of the 2 x 2 matrix which acts non-trivially on that qubit.
	# Each number in the support is mapped to a pair of alphabets in the characters list, as: x -> (characters[2x], characters[2x + 1]).
	# Eg. (x, y, z) ---> (C[2x] C[2y] C[2z] , C[2x + 1] C[2y + 1] C[2z + 1])
	n_interactions = interactions.shape[0]
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
	
	n_unique = Unique(qubits, n_qubits, None, 0)
	
	unique_qubits = np.zeros(n_unique, dtype = np.int)
	Unique(qubits, n_qubits, unique_qubits, 1)

	print("%d unique qubits\n{}" % (n_unique, unique_qubits))
	
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

	return free_index