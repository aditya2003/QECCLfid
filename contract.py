import string
import numpy as np
#import tensorflow as tf


def OptimalEinsum(scheme, ops, opt = "greedy"):
	# Contract a tensor network using einsum supplemented with its optimization tools.
	ops_args = ", ".join([("ops[%d]" % d) for d in range(len(ops))])
	#print("Calling np.einsum({}, {})\nwhere shapes are\n{}.".format(scheme, ops_args, [op.shape for op in ops]))
	path = eval("np.einsum_path(\'%s\', %s, optimize=\'%s\')" % (scheme, ops_args, opt))
	#print("Contraction process\n{}: {}\n{}".format(path[0][0], path[0][1:], path[1]))
	prod = np.einsum(scheme, *ops, optimize=path[0])
	# prod = tf.einsum(scheme, *[tf.convert_to_tensor(op) for op in ops], optimize=opt)
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


def ContractTensorNetwork(theta_dict, MAX = 10):
	# Compute the Theta matrix of a composition of channels.
	# The individual channels are provided a list where each one is a pair: (s, O) where s is the support and O is the theta matrix.
	# We will use einsum to contract the tensor network of channels.
	supports = [list(sup) for (sup, op) in theta_dict]
	if (len(supports) > MAX):
		partial_network = theta_dict[:MAX]
		partial_contraction = ContractThetaNetwork(partial_network)
		remaining_network = theta_dict[MAX:]
		remaining_contraction = ContractThetaNetwork(remaining_network)
		return ContractThetaNetwork(partial_contraction + remaining_contraction)
	(contraction_labels, free_labels) = SupportToLabel(supports)
	#print("contraction_labels = {}".format(contraction_labels))
	row_labels = ["".join([q[0] for q in interac]) for interac in contraction_labels]
	#print("row_contraction_labels = {}".format(row_labels))
	col_labels = ["".join([q[1] for q in interac]) for interac in contraction_labels]
	#print("col_contraction_labels = {}".format(col_labels))
	left = ["%s%s" % (row_labels[i], col_labels[i]) for i in range(len(contraction_labels))]
	#print("left = {}".format(left))
	contraction_scheme = "%s" % (",".join(left))
	#print("contraction_scheme = {}".format(contraction_scheme))
	free_row_labels = [free_labels[q][0] for q in free_labels]
	#print("free_row_labels = {}".format(free_row_labels))
	free_col_labels = [free_labels[q][1] for q in free_labels]
	#print("free_col_labels = {}".format(free_col_labels))
	contracted_labels = "%s%s" % ("".join(free_row_labels), "".join(free_col_labels))
	#print("contracted_labels = {}".format(contracted_labels))
	scheme = "%s->%s" % (contraction_scheme, contracted_labels)
	#print("Contraction scheme = {}".format(scheme))
	theta_ops = [op for (__, op) in theta_dict]
	composed = OptimalEinsum(scheme, theta_ops)
	composed_support = np.unique([q for (sup, op) in theta_dict for q in sup])
	composed_dict = [(composed_support, composed)]
	return composed_dict


if __name__ == '__main__':
	theta_dict = [(range(4), np.random.rand(4,4,4,4,4,4,4,4)), ((0,1), np.random.rand(4,4,4,4)), ((1,2), np.random.rand(4,4,4,4)), ((2,3), np.random.rand(4,4,4,4))]
	#print("Contracting the Theta network\n{}".format(theta_dict))
	(contracted_support, contracted_split) = ContractThetaNetwork(theta_dict, MAX=2)[0]
	print("Result supported on {} has dimensions {}.".format(contracted_support, contracted_split.shape))

	(contracted_support, contracted_all) = ContractThetaNetwork(theta_dict, MAX=4)[0]
	print("Result supported on {} has dimensions {}.".format(contracted_support, contracted_all.shape))
