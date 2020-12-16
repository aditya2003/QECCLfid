import numpy as np

def OptimalEinsum(scheme, ops, opt = "greedy"):
	# Contract a tensor network using einsum supplemented with its optimization tools.
	ops_args = ", ".join([("ops[%d]" % d) for d in range(len(ops))])
	path = eval("np.einsum_path(\'%s->\', %s, optimize=\'%s\')" % (scheme, ops_args, opt))
	print("Contraction process\n{}: {}\n{}".format(path[0][0], path[0][1:], path[1]))
	prod = np.einsum(scheme, *ops, optimize=path[0])
	return prod

def TensorTranspose(tensor):
	# Transpose the tensor, in other words, exchange its row and column indices.
	rank = int(tensor.ndim//2)
	tp_indices = np.array([(i, i + rank) for i in range(rank)], dtype=np.int).flatten()
	return np.transpose(tensor, tp_indices)

def GetNQubitPauli(ind, nq):
	# Compute the n-qubit Pauli that is at position 'i' in an ordering based on [I, X, Y, Z].
	# We will express the input number in base 4^n - 1.
	pauli = np.zeros(nq, dtype = np.int)
	for i in range(nq):
		pauli[i] = ind % 4
		ind = int(ind//4)
	return pauli

def TensorKron(tn1, tn2):
	# Compute the Kronecker product of two tensors A and B.
	# This is not equal to np.tensordot(A, B, axis = 0), see: https://stackoverflow.com/questions/52125078/why-does-tensordot-reshape-not-agree-with-kron.
	# We will implement this using einsum, as
	# np.einsum('i1 i2 .. in, j1 j2 ... jn -> i1 j1 i2 j2 ... in jn', A, B).
	if (tn1.ndim != tn2.ndim):
		print("TensorKron does not work presently for tensors of different dimensions.")
		return None
	tn1_inds = [("i%d" % i) for i in range(tn1.ndim)]
	tn2_inds = [("j%d" % i) for i in range(tn2.ndim)]
	kron_inds = ["%s%s" % (tn1_inds[i], tn2_inds[i]) for i in range(tn1.ndim)]
	scheme = ("%s, %s -> %s" % ("".join(tn1_inds), "".join(tn2_inds), "".join(kron_inds)))
	return OptimalEinsum(scheme, [tn1, tn2], opt = "greedy")


def TraceDot(tn1, tn2):
	# Compute the trace of the dot product of two tensors A and B.
	# If the indices of A are i_0 i_1, ..., i_(2n-1) and that of B are j_0 j_1 ... j_(2n-1)
	# then want to contract the indices i_(2k) with j_(2k+1), for all k in [0, n-1].
	# While calling np.einsum, we need to ensure that the row index of A is equal to the column index of B.
	# Additionally to ensure that we have a trace, we need to match the row and column indices of the product.
	tn1_inds = [("r%dc%d" % i) for i in range(tn1.ndim)]
	tn2_inds = [("c%dr%d" % i) for i in range(tn2.ndim)]
	scheme = ("%s, %s ->" % ("".join(tn1_inds), "".join(tn2_inds)))
	return OptimalEinsum(scheme, [tn1, tn2], opt = "greedy")

def KraussToTheta(kraus):
	# Convert from the Kraus representation to the "Theta" representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
	# X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
	# 	   = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
	# So we find that
	# T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	# We will store T as a Tensor with dimension = (2 * number of qubits) and bond dimension = 4.
	nq = kraus.shape[0]
	theta = np.array(*[4, 4]*nq, dtype = np.complex128)
	for i in range(4**nq):
		for j in range(4**nq):
			Pi = get_Pauli_tensor(GetNQubitPauli(i))
			Pj = GetNQubitPauli(j)
			PjT = TensorTranspose(get_Pauli_tensor(Pj))
			PioPjT = TensorKron(Pi, PjT)
			coeff = 0 + 0 * 1j
			for k in range(kraus.shape[0]):
				K = np.reshape(kraus[k, :, :], *[2, 2]*nq)
				Kdag = np.conj(TensorTranspose(K))
				coeff += TraceDot(Pi, K) * TraceDot(Pj, Kdag)
			theta += coeff * PioPjT
	return theta


def SupportToLabel(support, characters = None):
	# Convert a list of qubit indices to labels for a tensor.
	# Each qubit index corresponds to a pair of labels, indicating the row and column indices of the 2 x 2 matrix which acts non-trivially on that qubit.
	# Each number in the support is mapped to a pair of alphabets in the characters list, as: x -> (characters[2x], characters[2x + 1]).
	# Eg. (x, y, z) ---> (C[2x] C[2y] C[2z] , C[2x + 1] C[2y + 1] C[2z + 1])
	start = 0
	if characters == None:
		characters = string.printable
		start = 10
	label = ['0' for __ in range(2 * len(support))]
	nq = len(support)
	for s in range(nq):
		label[s] = characters[start + 2 * support[s]]
		label[nq + s] = characters[start + 2 * support[s]]
	return label

def ContractThetaNetwork(theta_dict):
	# Compute the Theta matrix of a composition of channels.
	# The individual channels are provided a list where each one is a pair: (s, O) where s is the support and O is the theta matrix.
	# We will use einsum to contract the tensor network of channels.
	supports = np.array([sup for (sup, op) in theta_dict], dtype = np.int).flatten()
	labels = ",".join(["".join(SupportToLabel(sup)) for (sup, op) in theta_dict])
	(__, order) = np.unique(supports, return_indices=True)
	composed_support = supports[np.argsort(order)]
	composed_label = "".join(SupportToLabel(composed_support))
	contraction_scheme = "%s->%s" % (labels, composed_label)
	theta_ops = [op for (__, op) in theta_dict]
	composed = OptimalEinsum(contraction_scheme, theta_ops)
	return composed