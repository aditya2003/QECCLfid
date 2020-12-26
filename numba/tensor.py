import numpy as np
import string
from define.QECCLfid.contract import OptimalEinsum

def TensorTrace(tensor, indices = "all", characters = None):
	# Compute the Trace of the tensor.
	if characters == None:
		characters = [c for c in string.ascii_lowercase] + [c for c in string.ascii_uppercase]
	labels = [characters[i] for i in range(tensor.ndim)]
	if indices == "all":
		indices = range(tensor.ndim//2)
	for i in indices:
		labels[i] = labels[i + int(tensor.ndim//2)]
	# Find unique labels in labels.
	print("labels = {}".format(labels))
	(right, counts) = np.unique(labels, return_counts=True)
	free_labels = list(right[np.argwhere(counts == 1).flatten()])
	scheme = "%s->%s" % ("".join(labels), "".join(free_labels))
	trace = OptimalEinsum(scheme, [tensor], opt="greedy", verbose=0)
	return trace

def TensorTranspose(tensor):
	# Transpose the tensor, in other words, exchange its row and column indices.
	# Note that when we reshape a matrix into (D, D, ..., D) tensor, it stores the indices as as
	# row_1, row_2, row_3, ..., row_(D/2), col_1, ..., col_(D/2).
	rows = range(0, tensor.ndim//2)
	cols = range(tensor.ndim//2, tensor.ndim)
	tp_indices = np.concatenate((cols, rows))
	return np.transpose(tensor, tp_indices)

def TraceDot(tn1, tn2):
	# Compute the trace of the dot product of two tensors A and B.
	# If the indices of A are i_0 i_1, ..., i_(2n-1) and that of B are j_0 j_1 ... j_(2n-1)
	# then want to contract the indices i_(2k) with j_(2k+1), for all k in [0, n-1].
	# While calling np.einsum, we need to ensure that the row index of A is equal to the column index of B.
	# Additionally to ensure that we have a trace, we need to match the row and column indices of the product.
	tn1_rows = [string.printable[10 + i] for i in range(tn1.ndim//2)]
	tn1_cols = [string.printable[10 + i] for i in range(tn1.ndim//2, tn1.ndim)]
	# The column indices of tn1 should match row indices of tn2
	# So, tn1_cols = tn2_rows.
	# the row and column indices of the product must match
	# So, tn1_rows = tn2_cols.
	scheme = ("%s%s,%s%s->" % ("".join(tn1_rows), "".join(tn1_cols), "".join(tn1_cols), "".join(tn1_rows)))
	return OptimalEinsum(scheme, [tn1, tn2], opt = "greedy")

def TensorKron(tn1, tn2):
	# Compute the Kronecker product of two tensors A and B.
	# This is not equal to np.tensordot(A, B, axis = 0), see: https://stackoverflow.com/questions/52125078/why-does-tensordot-reshape-not-agree-with-kron.
	# Note that when we reshape a matrix into (D, D, ..., D) tensor, it stores the indices as as
	# row_1, row_2, row_3, ..., row_(D/2), col_1, ..., col_(D/2).
	# We will implement Kronecker product using einsum, as
	# np.einsum('rA1 rA2 .. rAn, cA1 cA2 .. cAn, rB1 rB2 .. rBn, cB1 cB2 .. cBn -> rA1 rB1 rA2 rB2 .. rAn rBn, cA1 cB1 cA2 cB2 .. cAn cBn', A, B).
	if (tn1.ndim != tn2.ndim):
		print("TensorKron does not work presently for tensors of different dimensions.")
		return None
	tn1_rows = [string.printable[10 + i] for i in range(tn1.ndim//2)]
	tn1_cols = [string.printable[10 + i] for i in range(tn1.ndim//2, tn1.ndim)]
	tn2_rows = [string.printable[10 + tn1.ndim + i] for i in range(tn2.ndim//2)]
	tn2_cols = [string.printable[10 + tn1.ndim + i] for i in range(tn2.ndim//2, tn2.ndim)]
	#kron_inds = ["%s%s" % (tn1_rows[i], tn2_rows[i]) for i in range(tn1.ndim//2)]
	#kron_inds += ["%s%s" % (tn1_cols[i], tn2_cols[i]) for i in range(tn1.ndim//2)]
	kron_inds = ["%s" % (tn1_rows[i]) for i in range(tn1.ndim//2)]
	kron_inds += ["%s" % (tn2_rows[i]) for i in range(tn1.ndim//2)]
	kron_inds += ["%s" % (tn1_cols[i]) for i in range(tn1.ndim//2)]
	kron_inds += ["%s" % (tn2_cols[i]) for i in range(tn1.ndim//2)]
	scheme = ("%s%s,%s%s->%s" % ("".join(tn1_rows), "".join(tn1_cols), "".join(tn2_rows), "".join(tn2_cols), "".join(kron_inds)))
	return OptimalEinsum(scheme, [tn1, tn2], opt = "greedy")


if __name__ == '__main__':
	# Testing TensorTranspose
	nq = 2
	tensor = np.reshape(np.random.rand(2**nq, 2**nq), [2, 2]*nq)
	tp_tensor = TensorTranspose(tensor)
	#print("tensor\n{}\nand its transpose\n{}".format(tensor.reshape(2**dims, 2**dims), tp_tensor.reshape(2**dims, 2**dims)))
	print("A ?= A.T is {}".format(np.allclose(tensor.reshape(2**nq, 2**nq), tp_tensor.reshape(2**nq, 2**nq).T)))

	# Testing TensorKron
	nq = 4
	tn1 = np.reshape(np.random.rand(2**nq, 2**nq), [2, 2]*nq)
	tn2 = np.reshape(np.random.rand(2**nq, 2**nq), [2, 2]*nq)
	kn_tensor = TensorKron(tn1, tn2)
	#print("A\n{}\nB\n{}\nA o B = \n{}".format(tn1.reshape(2**nq, 2**nq), tn2.reshape(2**nq, 2**nq), kn_tensor.reshape(4**nq, 4**nq)))
	print("TensorKron(A, B) ?= A o B is {}".format(np.allclose(kn_tensor.reshape(4**nq, 4**nq), np.kron(tn1.reshape(2**nq, 2**nq), tn2.reshape(2**nq, 2**nq)))))
