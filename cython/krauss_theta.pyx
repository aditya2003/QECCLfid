cimport cython
import numpy as np
cimport numpy as np
from tracedot cimport TraceDot
from pauli cimport ComputePauliMatrix, ComputeLexicographicPaulis
ctypedef double complex complex128_t

cdef extern from "math.h":
	double pow(double a, double x) nogil

cdef extern from "complex.h":
	double complex I
	double complex conj(double complex x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int MultiplyScalar(complex128_t scalar, complex128_t [:, :] matrix, int rows, int cols):
	# Multiply the entries of a matrix with a scalar.
	cdef:
		int i, j
	for i in range(rows):
		for j in range(cols):
			matrix[i, j] = matrix[i, j] * scalar
	return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int Add(complex128_t [:, :] A, complex128_t [:, :] B, long rows, long cols):
	# Add two matrices and overwrite the second matrix with the sum.
	cdef:
		int i, j
	for i in range(rows):
		for j in range(cols):
			B[i, j] = B[i, j] + A[i, j]
	return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def KrausToTheta(complex128_t [:, :, :] kraus, int nq):
	# Convert from the Kraus representation to the "Theta" representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
	# X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
	# 	   = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
	# So we find that
	# T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	cdef:
		Py_ssize_t i, j, k, q
		long dim, n_pauli
	
	dim = long(pow(2, nq))
	n_pauli = long(pow(dim, 2))

	## Assign support of Kraus
	# supp_K = tuple([m for m in range(nq)])
	# cdef:
	# 	complex128_t [:, :, :] krausH = np.conj(np.transpose(kraus, [0, 2, 1]))
	
	# PjoPi = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)
	# PjToPi = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)

	# Preparing the Pauli operators.
	cdef:
		int [:] transpose_signs = np.zeros(n_pauli, dtype = np.int32)
		int [:, :] pauli_operators = np.zeros((n_pauli, nq), dtype = np.int32)
		complex128_t [:, :, :] pauli_matrices = np.zeros((n_pauli, dim, dim), dtype = np.complex128)
	
	ComputeLexicographicPaulis(nq, pauli_matrices, 1, transpose_signs, 1, pauli_operators)

	# Chi and Theta matrix
	cdef:
		int [:] pauli_op_2nq = np.zeros(2 * nq, dtype = np.int32)
		complex128_t [:, :] pauli_matrix_2nq = np.zeros((n_pauli, n_pauli), dtype = np.complex128)
		complex128_t [:, :] chi = np.zeros((n_pauli, n_pauli), dtype = np.complex128)
		complex128_t [:, :] theta = np.zeros((n_pauli, n_pauli), dtype = np.complex128)
		complex128_t tr_Pi_K, tr_Pj_Kdag, transpose_sign
		
	for i in range(4**nq):
		for j in range(4**nq):
			transpose_sign = transpose_signs[j] + I * 0
			# Compute the Chi matrix first.
			# Since Chi is Hermitian, we only need to really compute its upper diagonal.
			if (i <= j):
				for k in range(kraus.shape[0]):
					tr_Pi_K = TraceDot(pauli_matrices[i, :, :], kraus[k, :, :], dim, dim, 0)
					tr_Pj_Kdag = TraceDot(pauli_matrices[j, :, :], kraus[k, :, :], dim, dim, 1)
					
					chi[i, j] = chi[i, j] + tr_Pi_K * tr_Pj_Kdag
				
				chi[i, j] = chi[i, j]/n_pauli
			
			else:
				chi[i, j] = conj(chi[j, i])
			
			# if (i == j):
			# 	if ((np.real(chi[i, j]) <= -1E-14) or (np.abs(np.imag(chi[i, j])) >= 1E-14)):
			# 		print("Error: Chi[%d, %d] = %g + i %g" % (i, j, np.real(chi[i, j]), np.imag(chi[i, j])))
			# 		exit(0)
			# Compute the tensor product of two Pauli operators.
			for q in range(nq):
				pauli_op_2nq[q] = pauli_operators[i, q]
			for q in range(nq, 2 * nq):
				pauli_op_2nq[q] = pauli_operators[j, q - nq]
			ComputePauliMatrix(pauli_op_2nq, 2 * nq, pauli_matrix_2nq)
			MultiplyScalar(chi[i, j] * transpose_sign, pauli_matrix_2nq, n_pauli, n_pauli)
			Add(pauli_matrix_2nq, theta, n_pauli, n_pauli)
			
			# print("Chi[%d, %d] = %g + i %g was computed in %d seconds." % (i, j, np.real(chi[i, j]), np.imag(chi[i, j]), timer() - click))
		# print("----")
	
	# print("Theta matrix was computed in {} seconds.".format(timer() - click))
	return np.asarray(theta)