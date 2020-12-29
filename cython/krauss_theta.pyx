cimport cython
import numpy as np
cimport numpy as np
from contract import TraceNetwork
# from timeit import default_timer as timer
# int nq = int(np.log2(kraus.shape[1]))

@cython.boundscheck(False)
@cython.wraparound(False)
def KraussToTheta(complex [:, :, :] kraus, int nq):
	# Convert from the Kraus representation to the "Theta" representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
	# X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
	# 	   = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
	# So we find that
	# T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	# We will store T as a Tensor with dimension = (2 * number of qubits) and bond dimension = 4.
	cdef:
		Py_ssize_t i, j, k, q
		int index, y_count
		int transpose_sign
		complex tr_Pi_K, tr_Pi_K_dag

	# cdef:
	# 	np.ndarray[complex, ndim=3] Paulis = np.zeros((4, 2, 2), dtype = complex)
	# 	np.ndarray[complex, ndim=2] chi = np.zeros((4**nq, 4**nq), dtype = complex)
	# 	np.ndarray[int, ndim=1] supp_Kdag = np.arange(nq, dtype = int)
	# 	np.ndarray[int, ndim=1] supp_K = np.arange(nq, dtype = int) + nq
	# 	np.ndarray[int, ndim=1] transpose_signs = np.zeros(4**nq, dtype = int)
	# 	np.ndarray[int, ndim=2] pauli_operators = np.zeros((4**nq, nq), dtype = int)

	## Define Pauli matrices
	cdef complex[:, :, :] Paulis = np.zeros((4, 2, 2), dtype = complex)
	for q in range(4):
		for i in range(2):
			for j in range(2):
				Paulis[q, i, j] = 0 + 0 * 1j
	# Pauli matrix: I
	Paulis[0, 0, 0] = 1
	Paulis[0, 1, 1] = 1
	# Pauli matrix: X
	Paulis[1, 0, 1] = 1
	Paulis[1, 1, 0] = 1
	# Pauli matrix: Y
	Paulis[1, 0, 1] = 1j
	Paulis[1, 1, 0] = -1j
	# Pauli matrix: Z
	Paulis[1, 0, 0] = 1
	Paulis[1, 1, 1] = -1
		
	## Assign support of Kraus
	cdef int[:] supp_Kdag = np.arange(nq, dtype = int)
	cdef int[:] supp_K = nq + np.arange(nq, dtype = int)
	
	# Theta matrices
	theta = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = complex)
	PjoPi = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = complex)
	PjToPi = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = complex)

	# Preparing the Pauli operators.
	cdef int[:] transpose_signs = np.zeros(4**nq, dtype = int)
	cdef int[:, :] pauli_operators = np.zeros((4**nq, nq), dtype = int)
	# click = timer()
	for i in range(4**nq):
		index = i
		y_count = 0
		for q in range(nq):
			pauli_operators[i, nq - q - 1] = index % 4
			index = int(index//4)
			# Count the number of Ys' for computing the transpose.
			if (pauli_operators[i, q] == 2):
				y_count += 1
		transpose_signs[i] = (-1) ** y_count 
	

	# click = timer()
	cdef complex[:, :] chi = np.zeros((4**nq, 4**nq), dtype = complex)

	for i in range(4**nq):
		Pi_tensor = []
		for q in range(nq):
			Pi_tensor.append(((q,), Paulis[pauli_operators[i, q]]))
		# Pi_tensor = [((q,), Paulis[pauli_operators[i, q], :, :]) for q in range(nq)]
		
		for j in range(4**nq):
			Pj_tensor = []
			for q in range(nq):
				Pj_tensor.append(((q,), Paulis[pauli_operators[j, q]]))

			# click = timer()

			if (i <= j):
				chi[i, j] = 0 + 0 * 1j
				for k in range(kraus.shape[0]):
					K = [(supp_K, np.reshape(kraus[k, :, :], tuple([2, 2]*nq)))]
					Kdag = [(supp_Kdag, np.reshape(np.conj(kraus[k, :, :].T), tuple([2, 2]*nq)))]
					
					(__, tr_Pi_K) = TraceNetwork(K + Pi_tensor, end_trace=1)
					(__, tr_Pj_Kdag) = TraceNetwork(Kdag + Pj_tensor, end_trace=1)
					
					chi[i, j] += tr_Pi_K * tr_Pj_Kdag
				
				chi[i, j] /= 4**nq
			
			else:
				chi[i, j] = np.conj(chi[j, i])
			
			if (i == j):
				if ((np.real(chi[i, j]) <= -1E-14) or (np.abs(np.imag(chi[i, j])) >= 1E-14)):
					print("Error: Chi[%d, %d] = %g + i %g" % (i, j, np.real(chi[i, j]), np.imag(chi[i, j])))
					exit(0)
			
			# Compute the tensor product of two Pauli operators.
			# We simply need to load a Pauli operator on every pair of row and column axis of the tensor PjoPi.
			# Tensor product is defined using the operation:
			# A[r1, c1] o B[r2, c2] = A[r1 r2, c1 c2]
			# We will implement Kronecker product using einsum, as
			# np.einsum(A, [rA, cA], B, [rB, cB], C, [rC, cC], D, ..., [rA, rB, rC, ..., cA, cB, cC, ...])
			# np.einsum(A, [rA1 rA2 .. rAn, cA1 cA2 .. cAn], B, [rB1 rB2 .. rBn, cB1 cB2 .. cBn], [rA1 rB1 rA2 rB2 .. rAn rBn, cA1 cB1 cA2 cB2 .. cAn cBn]).
			scheme = []
			for q in range(nq):
				scheme.append(Paulis[pauli_operators[j, q]])
				scheme.append([2 * q, 2 * q + 1])
			for q in range(nq, 2 * nq):
				scheme.append(Paulis[pauli_operators[i, q]])
				scheme.append([2 * q, 2 * q + 1])
			scheme.append([2 * q for q in range(2 * nq)] + [2 * q + 1 for q in range(2 * nq)])
			
			# Using numpy einsum to compute the tensor product.
			PjoPi = np.einsum(scheme, optimize="greedy")
			
			PjToPi = PjoPi * transpose_signs[j]
			theta += chi[i, j] * PjToPi
			
			# print("Chi[%d, %d] = %g + i %g was computed in %d seconds." % (i, j, np.real(chi[i, j]), np.imag(chi[i, j]), timer() - click))
		# print("----")
	
	# print("Theta matrix was computed in {} seconds.".format(timer() - click))
	return theta

