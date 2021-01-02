import cython
#cimport cython as cc
import numpy as np
cimport numpy as np
# from timeit import default_timer as timer

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
		int [:, :] interactions = np.zeros((n_interactions, max_interaction_range + 1), dtype = np.int32)
	for i in range(n_interactions):
		interactions[i, 0] = len(network[i][0])
		for j in range(1, 1 + interactions[i, 0]):
			interactions[i, j] = network[i][0][j - 1]

	print("n_interactions = %d" % (n_interactions))
	print(np.asarray(interactions))

	cdef:
		Py_ssize_t n_free
		int [:, :, :] contraction_labels = np.zeros((n_interactions, max_interaction_range, 2), dtype = np.int32)
		int [:, :] free_labels = SupportToLabel(interactions, contraction_labels)
	n_free = free_labels.shape[0]

	# Arrange the row and column labels for the contraction indices.
	cdef:
		int [:, :] left = np.zeros((n_interactions, 2 * max_interaction_range), dtype = np.int32)
	
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