import string
import numpy as np
import numba as nb
from einsumt import einsumt as einsum_parallel
from timeit import default_timer as timer


def EinsumTrace(scheme, ops):
	# Contract a tensor network using einsum supplemented with its optimization tools.
	ops_args = ", ".join([("ops[%d]" % d) for d in range(len(ops))])
	# print("Calling np.einsum({}, {})\nwhere shapes are\n{}.".format(scheme, ops_args, [op.shape for op in ops]))
	start = timer()
	# path = eval("np.einsum_path(\'%s\', %s, optimize=\'%s\')" % (scheme, ops_args, opt))
	path = np.einsum_path(scheme, *ops, optimize=opt)
	if verbose == 1:
		print("Contraction process\n{}: {}\n{}".format(path[0][0], path[0][1:], path[1]))
	if (parallel == 0):
		prod = np.einsum(scheme, *ops, optimize=path[0])
	else:
		prod = einsum_parallel(scheme, *ops, optimize=path[0])
	# tfops = [tf.convert_to_tensor(op) for op in ops]
	# prod = tf.einsum(scheme, *tfops, optimize=opt)
	end = timer()
	if verbose == 1:
		print("Einsum({}, {})\nwhere shapes are\n{}\ntook {} seconds.".format(scheme, ops_args, [op.shape for op in ops], int(end - start)))
	return prod


def OptimalEinsum(scheme, ops, opt = "greedy", verbose=0, parallel=0):
	# Contract a tensor network using einsum supplemented with its optimization tools.
	ops_args = ", ".join([("ops[%d]" % d) for d in range(len(ops))])
	#print("Calling np.einsum({}, {})\nwhere shapes are\n{}.".format(scheme, ops_args, [op.shape for op in ops]))
	start = timer()
	# path = eval("np.einsum_path(\'%s\', %s, optimize=\'%s\')" % (scheme, ops_args, opt))
	path = np.einsum_path(scheme, *ops, optimize=opt)
	if verbose == 1:
		print("Contraction process\n{}: {}\n{}".format(path[0][0], path[0][1:], path[1]))
	if (parallel == 0):
		prod = np.einsum(scheme, *ops, optimize=path[0])
	else:
		prod = einsum_parallel(scheme, *ops, optimize=path[0])
	# tfops = [tf.convert_to_tensor(op) for op in ops]
	# prod = tf.einsum(scheme, *tfops, optimize=opt)
	end = timer()
	if verbose == 1:
		print("Einsum({}, {})\nwhere shapes are\n{}\ntook {} seconds.".format(scheme, ops_args, [op.shape for op in ops], int(end - start)))
	return prod


def SupportToLabel(supports, characters = None):
	# Convert a list of qubit indices to labels for a tensor.
	# Each qubit index corresponds to a pair of labels, indicating the row and column indices of the 2 x 2 matrix which acts non-trivially on that qubit.
	# Each number in the support is mapped to a pair of alphabets in the characters list, as: x -> (characters[2x], characters[2x + 1]).
	# Eg. (x, y, z) ---> (C[2x] C[2y] C[2z] , C[2x + 1] C[2y + 1] C[2z + 1])
	if characters == None:
		characters = [c for c in string.ascii_lowercase] + [c for c in string.ascii_uppercase]
	#print("characters\n{}".format(characters))
	#print("support\n{}".format(supports))
	labels = [[[-1, -1] for q in interac] for interac in supports]
	#print("labels\n{}".format(labels))
	unique_qubits = np.unique([q for sup in supports for q in sup])
	#print("unique qubits\n{}".format(unique_qubits))
	free_index = {q:[-1, -1] for q in unique_qubits}
	for i in range(len(supports)):
			sup = supports[i]
			#print("Support: {}".format(sup))
			for j in range(len(sup)):
					#print("Qubit: {}".format(sup[j]))
					q = sup[j]
					if (free_index[q][0] == -1):
						free_index[q][0] = characters.pop()
						free_index[q][1] = characters.pop()
						#print("Assigning {} and {} to qubit {} of map {}\n".format(free_index[q][0],free_index[q][1],q,i))
						labels[i][j][0] = free_index[q][0]
						labels[i][j][1] = free_index[q][1]
					else:
						labels[i][j][0] = free_index[q][1]
						free_index[q][1] = characters.pop()
						labels[i][j][1] = free_index[q][1]
						#print("Assigning {} and {} to qubit {} of map {}\n".format(labels[i][j][0],labels[i][j][1],q,i))
					#print("labels\n{}\nfree index\n{}".format(labels, free_index))
	#print("labels\n{}\nfree index\n{}".format(labels, free_index))
	return (labels, free_index)


@nb.guvectorize(["void(complex128[:, :, :], uint16, uint16, complex128[:])"], "(k,n,n),(),()->()")
@nb.njit("complex128(uint16[:,:], uint8, complex128[:])")
def TraceNetwork(supports, bond_dimension, operators):
	# Contract a tensor network, for the special case where the result is a scalar.
	# The supports is specified as a numpy array, each of whose rows contain the support for every operator.
	# Each row of the supports array contains the size of the support followed by the support qubits.
	supports_list = [supports[i, 1 : supports[i, 0]] for i in range(supports.shape[0])]
	operators_list = []
	start = 0
	for i in range(supports.shape[0]):
		end = start + bond_dimension ** (2 * supports[i, 0])
		op_shape = np.array([bond_dimension for q in range(supports[i, 0]) for d in range(2)], dtype = nb.int64)
		op = np.ascontiguousarray(operators[start : end]).reshape(op_shape)
		operators_list.append(op)
		start = end

	# Convert a list of qubit indices to labels for a tensor.
	characters = [c for c in string.ascii_lowercase] + [c for c in string.ascii_uppercase]
	
	contraction_labels = [[[-1, -1] for q in interac] for interac in supports_list]
	unique_qubits = np.unique([q for sup in supports_list for q in sup])
	
	# free_labels = {q:[-1, -1] for q in unique_qubits}
	free_labels = [[-1, -1] for q in range(np.max(unique_qubits))]

	for i in range(len(supports_list)):
		sup = supports_list[i]
		for j in range(len(sup)):
			# print("Qubit: {}".format(sup[j]))
			q = sup[j]
			if (free_labels[q][0] == -1):
				free_labels[q][0] = characters.pop()
				free_labels[q][1] = characters.pop()
				#print("Assigning {} and {} to qubit {} of map {}\n".format(free_labels[q][0],free_labels[q][1],q,i))
				contraction_labels[i][j][0] = free_labels[q][0]
				contraction_labels[i][j][1] = free_labels[q][1]
			else:
				contraction_labels[i][j][0] = free_labels[q][1]
				free_labels[q][1] = characters.pop()
				contraction_labels[i][j][1] = free_labels[q][1]
				#print("Assigning {} and {} to qubit {} of map {}\n".format(labels[i][j][0],labels[i][j][1],q,i))
			#print("labels\n{}\nfree index\n{}".format(labels, free_labels))
	#print("labels\n{}\nfree index\n{}".format(labels, free_labels))
	
	# print("contraction_labels = {}".format(contraction_labels))
	row_labels = ["".join([q[0] for q in interac]) for interac in contraction_labels]
	# print("row_contraction_labels = {}".format(row_labels))
	col_labels = ["".join([q[1] for q in interac]) for interac in contraction_labels]
	# print("col_contraction_labels = {}".format(col_labels))
	left = ",".join(["%s%s" % (row_labels[i], col_labels[i]) for i in range(len(contraction_labels))])
	# print("left = {}".format(left))
	# free_row_labels = [free_labels[q][0] for q in free_labels]
	free_row_labels = [free_labels[q][0] for q in unique_qubits]
	#print("free_row_labels = {}".format(free_row_labels))
	# free_col_labels = [free_labels[q][1] for q in free_labels]
	free_col_labels = [free_labels[q][1] for q in unique_qubits]
	#print("free_col_labels = {}".format(free_col_labels))
	
	# If the last operation is a trace, we need to contract the free row and column indices.
	# So we should make sure that the i-th free row index = i-th free column index.
	# print("left before end trace\n{}-->{}|{}".format(left, "".join(free_row_labels), "".join(free_col_labels)))
	for (r_lab, c_lab) in zip(free_row_labels, free_col_labels):
		left = left.replace(c_lab, r_lab)
	# print("left after end trace = {}".format(left))
	
	# Optimal einsum
	path = np.einsum_path("%s->" % (left), operators_list, optimize="greedy")
	trace = np.einsum("%s->" % (left), operators_list, optimize=path[0])
	return trace


def ContractTensorNetwork(network, end_trace=0, parallel=0):
	# Compute the Theta matrix of a composition of channels.
	# The individual channels are provided a list where each one is a pair: (s, O) where s is the support and O is the theta matrix.
	# We will use einsum to contract the tensor network of channels.
	# print("Function: ContractTensorNetwork")
	supports = [list(sup) for (sup, op) in network]

	# print("supports\n{}".format(supports))

	(contraction_labels, free_labels) = SupportToLabel(supports)
	# print("contraction_labels = {}".format(contraction_labels))
	row_labels = ["".join([q[0] for q in interac]) for interac in contraction_labels]
	# print("row_contraction_labels = {}".format(row_labels))
	col_labels = ["".join([q[1] for q in interac]) for interac in contraction_labels]
	# print("col_contraction_labels = {}".format(col_labels))
	left = ",".join(["%s%s" % (row_labels[i], col_labels[i]) for i in range(len(contraction_labels))])
	# print("left = {}".format(left))
	free_row_labels = [free_labels[q][0] for q in free_labels]
	#print("free_row_labels = {}".format(free_row_labels))
	free_col_labels = [free_labels[q][1] for q in free_labels]
	#print("free_col_labels = {}".format(free_col_labels))
	
	if end_trace == 1:
		# If the last operation is a trace, we need to contract the free row and column indices.
		# So we should make sure that the i-th free row index = i-th free column index.
		# print("left before end trace\n{}-->{}|{}".format(left, "".join(free_row_labels), "".join(free_col_labels)))
		for (r_lab, c_lab) in zip(free_row_labels, free_col_labels):
			left = left.replace(c_lab, r_lab)
		# print("left after end trace = {}".format(left))
		right = ""
		composed_support = []
	else:
		right = "%s%s" % ("".join(free_row_labels), "".join(free_col_labels))
		composed_support = np.unique([q for (sup, op) in network for q in sup])
	#print("right = {}".format(right))
	scheme = "%s->%s" % (left, right)
	# print("Contraction scheme = {}".format(scheme))
	theta_ops = [op for (__, op) in network]
	composed = OptimalEinsum(scheme, theta_ops, opt="greedy", verbose=0, parallel=0)
	#composed_dict = [(composed_support, composed)]
	return (composed_support, composed)


if __name__ == '__main__':
	theta_dict = [(range(4), np.random.rand(4,4,4,4,4,4,4,4)), ((0,1), np.random.rand(4,4,4,4)), ((1,2), np.random.rand(4,4,4,4)), ((2,3), np.random.rand(4,4,4,4))]
	#print("Contracting the Theta network\n{}".format(theta_dict))
	(contracted_support, contracted_split) = ContractThetaNetwork(theta_dict, MAX=2)[0]
	print("Result supported on {} has dimensions {}.".format(contracted_support, contracted_split.shape))

	(contracted_support, contracted_all) = ContractThetaNetwork(theta_dict, MAX=4)[0]
	print("Result supported on {} has dimensions {}.".format(contracted_support, contracted_all.shape))