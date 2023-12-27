import numpy as np


def Unique(arr):
	# Compute the number of unique elements in an integer array.
	print("searching for unique elements in\n{}".format(arr))
	size = arr.shape[0]
	unique_indicator = np.ones(size, dtype = np.int64)
	for i in range(size):
		for j in range(i + 1, size):
			if (arr[i] == arr[j]):
				unique_indicator[j] = 0

	# Count the unique elements
	n_unique = np.sum(unique_indicator)

	print("{} unique elements\n{}".format(n_unique, unique_indicator))

	# Load the unique elements in the array
	unique = np.zeros(n_unique, dtype = np.int64)
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
	qubits = [q for interac in interactions for q in interac]
	# print("qubits\n{}".format(qubits))
	# Compute the unique qubits in the list of interactions.
	unique_qubits = np.unique(qubits)
	n_unique = unique_qubits.shape[0]

	# Compute the contraction labels and free indices according to the interactions.
	free_indices = {q: [-1, -1] for q in unique_qubits}
	contraction_labels = [[[-1, -1] for q in interac] for interac in interactions]
	symbols = list(range(2 * len(qubits)))
	n_symbols = 0
	for (i, interac) in enumerate(interactions):
		for (j, q) in enumerate(interac):
			if (free_indices[q][0] == -1):
				free_indices[q][0] = symbols[n_symbols]
				n_symbols += 1
				free_indices[q][1] = symbols[n_symbols]
				n_symbols += 1
				contraction_labels[i][j][0] = free_indices[q][0]
				contraction_labels[i][j][1] = free_indices[q][1]
			else:
				contraction_labels[i][j][0] = free_indices[q][1]
				free_indices[q][1] = symbols[n_symbols]
				n_symbols += 1
				contraction_labels[i][j][1] = free_indices[q][1]

	return (contraction_labels, free_indices, unique_qubits)
