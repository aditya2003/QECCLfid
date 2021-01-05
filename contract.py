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
	interactions = [sup for (sup, __) in network]
	# print("{} interactions\n{}".format(len(interactions), interactions))

	# Compute the contraction labels and the free labels.
	(contraction_labels, free_labels, qubits) = SupportToLabel(interactions)
	# print("contraction labels\n{}\nfree labels\n{}".format(contraction_labels, free_labels))

	# Arrange the operators to be contracted as row and column pairs.
	left = []
	for i in range(len(interactions)):
		interaction_range = len(interactions[i])
		labels = [-1 for __ in range(2 * interaction_range)]
		for j in range(interaction_range):
			labels[j] = contraction_labels[i][j][0] # Row label.
			labels[j + interaction_range] = contraction_labels[i][j][1] # Column label.
		left.append(labels)

	# print("left\n{}".format(left))

	if (end_trace == 1):
		# When the last operation is a trace, we need to contract the free row and column indices.
		# So we should make sure that the i-th free row index = i-th free column index.
		# Every free column index that appears in the left, must be replaced by the corresponding free row index.
		n_interactions = len(interactions)
		for q in free_labels:
			row_free = free_labels[q][0]
			col_free = free_labels[q][1]
			for i in range(len(left)):
				for j in range(len(left[i])):
					if (left[i][j] == col_free):
						left[i][j] = row_free
		# print("left after trace\n{}".format(left))
	else:
		# order the indices of the contracted network.
		right = [free_labels[q][0] for q in qubits] + [free_labels[q][1] for q in qubits]

	# Prepare the input to numpy's einsum
	scheme = []
	for i in range(len(network)):
		scheme.append(network[i][1])
		scheme.append(left[i])

	if (end_trace == 0):
		scheme.append(right)
		# print("contraction labels\n{}\nfree labels\n{}".format(contraction_labels, free_labels))
		# print("left\n{}".format(left))
		# print("right\n{}".format(right))

	# Contract the network using einsum
	# start = timer()
	contracted_support = tuple([q for q in qubits])
	contracted_operator = np.einsum(*scheme, optimize="greedy")
	contracted_network = (contracted_support, contracted_operator)
	# print("Contraction was done in %.3f seconds." % (timer() - click))
	return contracted_network
