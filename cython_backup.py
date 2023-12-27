##########################################################
# unique.pyx
##########################################################

ctypedef double complex complex128_t

#@cc.boundscheck(False) # Deactivate bounds checking
#@cc.wraparound(False) # Deactivate negative indexing.
cdef int Unique(int [:] arr, int size, int [:] unique, int return_elements):
	# Compute the number of unique elements in an integer array.
	
	print("searching for unique elements in")
	print(np.asarray(arr))

	cdef:
		Py_ssize_t i, j
		int [:] unique_indicator = np.ones(size, dtype = np.int6432)
	for i in range(size):
		for j in range(i + 1, size):
			if (arr[i] == arr[j]):
				unique_indicator[j] = 0
	
	# Count the unique elements
	cdef:
		int n_unique
	n_unique = np.sum(unique_indicator)

	print("%d unique elements" % (n_unique))
	print("unique_indicator")
	print(np.asarray(unique_indicator))
	
	if (return_elements == 0):
		# Load the unique elements in the array
		n_unique = 0
		for i in range(size):
			if (unique_indicator[i] == 1):
				unique[n_unique] = i
				n_unique += 1

	return n_unique

#@cc.boundscheck(False) # Deactivate bounds checking
#@cc.wraparound(False) # Deactivate negative indexing.
cdef int [:, :] SupportToLabel(int [:, :] interactions, int [:, :, :] contraction_labels):
	# Convert a list of qubit indices to labels for a tensor.
	# Each qubit index corresponds to a pair of labels, indicating the row and column indices of the 2 x 2 matrix which acts non-trivially on that qubit.
	# Each number in the support is mapped to a pair of alphabets in the characters list, as: x -> (characters[2x], characters[2x + 1]).
	# Eg. (x, y, z) ---> (C[2x] C[2y] C[2z] , C[2x + 1] C[2y + 1] C[2z + 1])
	cdef:
		Py_ssize_t i, j, q, n_symbols, n_interactions
	
	n_interactions = interactions.shape[0]
	n_symbols = 0
	for q in range(n_interactions):
		n_symbols += 2 * interactions[q, 0]
	cdef:
		int [:] symbols = np.arange(n_symbols, dtype = np.int6432)

	print("symbols")
	print(np.asarray(symbols))

	# Compute the unique qubits in the list of interactions.
	cdef:
		int [:] qubits = np.zeros(n_symbols//2, dtype = np.int6432)
		Py_ssize_t n_qubits
	n_qubits = 0
	for i in range(n_interactions):
		for q in range(1, 1 + interactions[i, 0]):
			qubits[n_qubits] = interactions[i, q]
			n_qubits += 1
	
	cdef:
		Py_ssize_t n_unique
		int [1] empty = [0]
	n_unique = Unique(qubits, n_qubits, empty, 0)
	
	cdef:
		int [:] unique_qubits = np.zeros(n_unique, dtype = np.int6432)
	Unique(qubits, n_qubits, unique_qubits, 1)

	print("%d unique qubits" % (n_unique))
	print(np.asarray(unique_qubits))

	# Free indices
	cdef:
		int [:, :] free_index = np.zeros((n_unique, 2), dtype = np.int6432)
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


##########################################################
# contract.pyx
##########################################################

def TraceNetwork(network):
	# Compute the trace of a tensor network.
	print("network")
	print(network)

	cdef:
		Py_ssize_t i, j, n_interactions, max_interaction_range

	n_interactions = len(network)
	
	max_interaction_range = 0
	for i in range(n_interactions):
		if (max_interaction_range < len(network[i][0])):
			max_interaction_range = len(network[i][0])
	
	cdef:
		int [:, :] interactions = np.zeros((n_interactions, max_interaction_range + 1), dtype = np.int6432)
	for i in range(n_interactions):
		interactions[i, 0] = len(network[i][0])
		for j in range(1, 1 + interactions[i, 0]):
			interactions[i, j] = network[i][0][j - 1]

	print("n_interactions = %d" % (n_interactions))
	print(np.asarray(interactions))

	cdef:
		Py_ssize_t n_free
		int [:, :, :] contraction_labels = np.zeros((n_interactions, max_interaction_range, 2), dtype = np.int6432)
		int [:, :] free_labels = SupportToLabel(interactions, contraction_labels)
	n_free = free_labels.shape[0]

	# Arrange the row and column labels for the contraction indices.
	cdef:
		int [:, :] left = np.zeros((n_interactions, 2 * max_interaction_range), dtype = np.int6432)
	
	print("contraction_labels")
	print(np.asarray(contraction_labels))
	print("free_labels")
	print(np.asarray(free_labels))

	for i in range(n_interactions):
		for j in range(interactions[i, 0]):
			# Row labels
			left[i, j] = contraction_labels[i, j, 0]
			# Column labels
			left[i, max_interaction_range + j] = contraction_labels[i, j, 1]

	print("left")
	print(np.asarray(left))

	# The last operation is a trace, we need to contract the free row and column indices.
	# So we should make sure that the i-th free row index = i-th free column index.
	# Every free column index that appears in the left, must be replaced by the corresponding free row index.
	cdef Py_ssize_t row_free, col_free
	for i in range(n_free):
		row_free = free_labels[i, 0]
		col_free = free_labels[i, 1]
		for j in range(n_interactions):
			for q in range(interactions[i, 0]):
				if (left[j, q + max_interaction_range] == col_free):
					left[j, q + max_interaction_range] = row_free

	print("left after trace")
	print(np.asarray(left))

	### Pure python code.
	# Prepare the input to numpy's einsum
	scheme = []
	for i in range(n_interactions):
		scheme.append(network[i][1])
		rows = left[i, :interactions[i, 0]]
		cols = left[i, max_interaction_range : (max_interaction_range + interactions[i, 0])]
		scheme.append(list(np.concatenate((rows, cols))))
	
	print("scheme")
	print(scheme)

	# einsum_scheme = ",".join(["scheme[%d]" % (m) for m in range(2 * n_interactions)])
	# print("einsum_scheme")
	# print(einsum_scheme)
	# for i in range(n_interactions):
	# 	print(np.round(scheme[2*i], 2))
	# 	print(scheme[2*i + 1])

	# Contract the network using einsum
	# start = timer()
	cdef:
		complex trace
	trace = np.einsum(*scheme, optimize="greedy")
	# trace = cython.inline("np.einsum(%s, optimize = \"greedy\")" % (einsum_scheme))
	# end = timer()

	# print("Einsum({}, {})\nwhere shapes are\n{}\ntook {} seconds.".format(scheme, ops_args, [op.shape for op in ops], int(end - start)))

	return trace