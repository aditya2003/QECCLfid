import cython
import numpy as np
from ncon import ncon
# import cotengra as ctg
from timeit import default_timer as timer
from define.QECCLfid.unique import SupportToLabel
# from unique import SupportToLabel # only for decoding purposes.

def OptimalEinsum(scheme, ops, opt = "greedy", verbose=0):
	# Contract a tensor network using einsum supplemented with its optimization tools.
	start = timer()
	prod = np.einsum(scheme, *ops, optimize="greedy")
	end = timer()
	if verbose == 1:
		print("Einsum({})\nwhere shapes are\n{}\ntook {} seconds.".format(scheme, [op.shape for op in ops], int(end - start)))
	return prod


def ContractTensorNetwork(network, end_trace=0, use_einsum=1):
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
		right = []
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
	if (max_tensor_index(left, right) >= 52):
		use_einsum = 0

	if (use_einsum == 1):
		contracted_operator = np.einsum(*scheme, optimize="greedy")
	else:
		operators = [op for (__, op) in network]
		contracted_operator = NCon(operators, left, right)
	contracted_network = (contracted_support, contracted_operator)
	# print("Contraction was done in %.3f seconds using %s." % (timer() - start, ["ncon", "numpy"][use_einsum]))
	return contracted_network

def max_tensor_index(contraction_schedule, free_labels):
	# Compute the maximum tensor index in the tensor network.
	tensor_labels = np.unique([l for axes_labels in contraction_schedule for l in axes_labels] + free_labels)
	max_label = np.max(tensor_labels)	
	return max_label

def NCon(operators, contraction_schedule, free_labels):
	# Contract a tensor network using ncon.
	# https://github.com/mhauru/ncon
	# ncon reserves positive valued axes labels for the contraction indices and negetive valued axes labels for the free indices.
	# Hence, we will adopt the following procedure to cast the indices into a form that is acceptable by ncon.
	# 1. Increment all the indices by 1.
	# 2. For all indices in the contraction schedule that also appear in free indices, negate their values.
	
	contraction_schedule_ncon = [[l + 1 for l in axes_labels] for axes_labels in contraction_schedule]
	free_labels_ncon = [fl + 1 for fl in free_labels]
	
	# print("contraction schedule = {}".format(tuple(contraction_schedule_ncon)))
	# print("free labels = {}".format(free_labels_ncon))

	for als in range(len(contraction_schedule_ncon)):
		axes_labels = contraction_schedule_ncon[als]
		for l in range(len(axes_labels)):
			if (axes_labels[l] in free_labels_ncon):
				contraction_schedule_ncon[als][l] *= -1
	free_labels_ncon = [-1 * free_labels_ncon[l] for l in range(len(free_labels_ncon))]
	
	# print("operators = {}".format(tuple(operators)))
	# print("contraction schedule ncon = {}".format(tuple(contraction_schedule_ncon)))
	# print("free labels ncon = {}".format(free_labels_ncon))
	result = ncon(operators, tuple(contraction_schedule_ncon), forder=free_labels_ncon)
	return result

def Cotengra(operators, contraction_schedule, free_labels):
	# Contract a tensor network using Cotegra.
	# https://cotengra.readthedocs.io/en/latest/basics.html
	labels_cotengra = list(map(ctg.get_symbol, np.unique([lab for interac in contraction_schedule for lab in interac] + free_labels)))
	
	contraction_schedule_cotengra = [list(map(ctg.get_symbol, interac)) for interac in contraction_schedule]
	free_labels_cotengra = list(map(ctg.get_symbol, free_labels))
	size_dict = {}
	for (i, interac) in enumerate(contraction_schedule_cotengra):
	    for a in range(len(interac)):
	    	size_dict[ctg.get_symbol(interac[a])] = operators[i].shape[a]

	opt = ctg.HyperOptimizer()
	tree = opt.search(contraction_schedule_cotengra, free_labels_cotengra, size_dict)
	result = tree.contract(operators)

	return result

if __name__ == '__main__':
	# We want to test the contraction tool for tensor networks.
	A = np.random.randn(2, 2, 2, 2)
	B = np.random.randn(2, 2)
	C = np.random.randn(2, 2, 2, 2)
	D = np.random.randn(2, 2, 2, 2, 2, 2)
	E = np.random.randn(2, 2, 2, 2)
	F = np.random.randn(2, 2, 2, 2, 2, 2)
	G = np.random.randn(2, 2, 2, 2, 2, 2)
	H = np.random.randn(2, 2, 2, 2, 2, 2, 2, 2)
	I = np.random.randn(2, 2)
	
	network = [((0,1), A), ((1,), B), ((1,2), C), ((0,2,3), D), ((1,3), E), ((1,2,3), F), ((0,1,3), G), ((0,1,2,3), H), ((3,), I)]
	
	(contracted_support, contracted_operator) = ContractTensorNetwork(network, end_trace=1, use_einsum=1)
	print("Contracted Tensor Network on Einsum is supported on {}:\n{}.".format(contracted_support, contracted_operator))

	(contracted_support, contracted_operator) = ContractTensorNetwork(network, end_trace=1, use_einsum=0)
	print("Contracted Tensor Network on NCon is supported on {}:\n{}.".format(contracted_support, contracted_operator))