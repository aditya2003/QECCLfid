cimport cython
import numpy as np
cimport numpy as np
# from timeit import default_timer as timer
ctypedef np.int32_t C_INT

@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
cdef int Unique(int [:] arr, int size, int [:] unique, int return_elements):
	# Compute the number of unique elements in an integer array.
	cdef:
		Py_ssize_t i, j
		int [:] unique_indicator = np.ones(size, dtype = int)
	for i in range(size):
		for j in range(size):
			if (i != j):
				if (arr[i] == arr[j]):
					unique_indicator[j] = 0
	
	# Count the unique elements
	cdef:
		int n_unique
	n_unique = np.sum(unique_indicator)
	
	if (return_elements == 0):
		# Load the unique elements in the array
		n_unique = 0
		for i in range(size):
			if (unique_indicator[i] == 1):
				unique[n_unique] = i
				n_unique += 1

	return n_unique

@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
cdef int [:, :] SupportToLabel(int [:, :] supports, int [:, :, :] contraction_labels):
	# Convert a list of qubit indices to labels for a tensor.
	# Each qubit index corresponds to a pair of labels, indicating the row and column indices of the 2 x 2 matrix which acts non-trivially on that qubit.
	# Each number in the support is mapped to a pair of alphabets in the characters list, as: x -> (characters[2x], characters[2x + 1]).
	# Eg. (x, y, z) ---> (C[2x] C[2y] C[2z] , C[2x + 1] C[2y + 1] C[2z + 1])
	cdef:
		Py_ssize_t i, j, q, n_symbols, n_supports, max_support_size
	
	n_supports = supports.shape[0]
	max_support_size = supports.shape[1]
	n_symbols = 0
	for q in range(n_supports):
		n_symbols += 2 * supports[q, 0]
	cdef:
		int [:] symbols = np.arange(n_symbols, dtype = np.int32)

	# Compute the unique qubits in the list of interactions.
	cdef:
		int [:] qubits = np.zeros(n_symbols//2, dtype = np.int32)
		Py_ssize_t n_qubits
	n_qubits = 0
	for i in range(n_supports):
		for q in range(1, 1 + supports[i, 0]):
			qubits[n_qubits] = supports[i, q]
			n_qubits += 1
	
	cdef:
		Py_ssize_t n_unique
		int [1] empty = [0]
	n_unique = Unique(qubits, n_qubits, empty, 0)
	
	cdef:
		int [:] unique_qubits = np.zeros(n_unique, dtype = np.int32)
	Unique(qubits, n_qubits, unique_qubits, 1)

	# Free indices
	cdef:
		int [:, :] free_index = np.zeros((n_unique, 2), dtype = np.int32)
	for q in range(n_unique):
		free_index[q, 0] = -1
		free_index[q, 1] = -1
	
	# Initialize contraction labels
	for i in range(n_supports):
		for q in range(1, 1 + supports[i, 0]):
			contraction_labels[i, q, 0] = -1
			contraction_labels[i, q, 1] = -1

	# Compute the contraction labels and free indices according to the interactions.
	for i in range(n_supports):
		
		for j in range(supports[i, 0]):
			
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


def TraceNetwork(network):
	# Compute the trace of a tensor network.
	cdef:
		Py_ssize_t i, j, n_supports, max_support_size

	n_supports = len(network)
	
	max_support_size = 0
	for i in range(n_supports):
		if (max_support_size < len(network[i][0])):
			max_support_size = len(network[i][0])
	
	cdef:
		int [:, :] supports = np.zeros((n_supports, max_support_size + 1), dtype = np.int32)
	for i in range(n_supports):
		supports[i, 0] = len(network[i][0])
		for j in range(1, 1 + supports[i, 0]):
			supports[i, j] = network[i][0][j - 1]

	cdef:
		Py_ssize_t n_free
		int [:, :, :] contraction_labels = np.zeros((n_supports, max_support_size, 2), dtype = np.int32)
		int [:, :] free_labels = SupportToLabel(supports, contraction_labels)
	n_free = free_labels.shape[0]

	# Arrange the row and column labels for the contraction indices.
	cdef:
		int [:, :] left = np.zeros((n_supports, 2 * max_support_size))
	for i in range(n_supports):
		for j in range(supports[i, 0]):
			# Row labels
			left[i, j] = contraction_labels[i, j, 0]
			# Column labels
			left[i, j + max_support_size] = contraction_labels[i, max_support_size + j, 1]

	# The last operation is a trace, we need to contract the free row and column indices.
	# So we should make sure that the i-th free row index = i-th free column index.
	# Every free column index that appears in the left, must be replaced by the corresponding free row index.
	cdef Py_ssize_t row_free, col_free
	for i in range(n_free):
		row_free = free_labels[i, 0]
		col_free = free_labels[i, 1]
		for j in range(n_supports):
			for q in range(supports[i, 0]):
				if (left[i, j + max_support_size] == col_free):
					left[i, j + max_support_size] = row_free

	### Pure python code.
	# Prepare the input to numpy's einsum
	operators = [op for (__, op) in network]
	scheme = []
	for i in range(n_supports):
		rows = left[i, :supports[i, 0]]
		cols = left[i, max_support_size : (max_support_size + supports[i, 0])]
		scheme.append(np.concatenate((rows, cols)))
		scheme.append(operators[i])
	scheme.append([])

	# Contract the network using einsum
	# start = timer()
	trace = np.einsum(scheme, optimize="greedy")
	# end = timer()

	# print("Einsum({}, {})\nwhere shapes are\n{}\ntook {} seconds.".format(scheme, ops_args, [op.shape for op in ops], int(end - start)))

	return trace