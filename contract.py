import cython
import numpy as np
from timeit import default_timer as timer
from define.QECCLfid.unique import SupportToLabel

def OptimalEinsum(scheme, ops, opt = "greedy", verbose=0, parallel=0):
	# Contract a tensor network using einsum supplemented with its optimization tools.
	#print("Calling np.einsum({}, {})\nwhere shapes are\n{}.".format(scheme, ops_args, [op.shape for op in ops]))
	start = timer()
	prod = np.einsum(scheme, *ops, optimize="greedy")
	end = timer()
	if verbose == 1:
		print("Einsum({}, {})\nwhere shapes are\n{}\ntook {} seconds.".format(scheme, ops_args, [op.shape for op in ops], int(end - start)))
	return prod


def ContractTensorNetwork(network, end_trace=0):
	# Compute the trace of a tensor network.
	print("network")
	print(network)

	n_interactions = len(network)
	
	# Compute the maximum range of interactions.
	max_interaction_range = 0
	for i in range(n_interactions):
		if (max_interaction_range < len(network[i][0])):
			max_interaction_range = len(network[i][0])
	
	# The interactions are stored as a numpy array.
	interactions = np.zeros((n_interactions, max_interaction_range + 1), dtype = np.int)
	for i in range(n_interactions):
		interactions[i, 0] = len(network[i][0])
		for j in range(1, 1 + interactions[i, 0]):
			interactions[i, j] = network[i][0][j - 1]

	print("n_interactions = %d\n{}" % (n_interactions, interactions))

	# Compute the contraction labels and the free labels.
	contraction_labels = np.zeros((n_interactions, max_interaction_range, 2), dtype = np.int)
	free_labels = SupportToLabel(interactions, contraction_labels)
	print("contraction_labels\n{}\nfree labels\n{}".format(contraction_labels, free_labels))
	
	# Arrange the contraction labels as row, column pairs.
	left = np.zeros((n_interactions, 2 * max_interaction_range), dtype = np.int)
	for i in range(n_interactions):
		for j in range(interactions[i, 0]):
			# Row labels
			left[i, j] = contraction_labels[i, j, 0]
			# Column labels
			left[i, max_interaction_range + j] = contraction_labels[i, j, 1]

	print("left\n{}".format(left))
	
	if (end_trace == 1):
		# The last operation is a trace, we need to contract the free row and column indices.
		# So we should make sure that the i-th free row index = i-th free column index.
		# Every free column index that appears in the left, must be replaced by the corresponding free row index.
		n_free = free_labels.shape[0]
		for i in range(n_free):
			row_free = free_labels[i, 0]
			col_free = free_labels[i, 1]
			for j in range(n_interactions):
				for q in range(interactions[i, 0]):
					if (left[j, q + max_interaction_range] == col_free):
						left[j, q + max_interaction_range] = row_free

		print("left after trace\n{}".format(left))
	
	# Prepare the input to numpy's einsum
	scheme = []
	for i in range(n_interactions):
		scheme.append(network[i][1])
		rows = left[i, :interactions[i, 0]]
		cols = left[i, max_interaction_range : (max_interaction_range + interactions[i, 0])]
		scheme.append(list(np.concatenate((rows, cols))))
	
	if (end_trace == 0):
		scheme.append(list(np.concatenate((free_labels[:, 0], free_labels[:, 1]))))

	print("scheme")
	print(scheme)

	# Contract the network using einsum
	# start = timer()
	trace = np.einsum(*scheme, optimize="greedy")
	# print("Contraction was done in %.3f seconds." % (timer() - click))
	return trace