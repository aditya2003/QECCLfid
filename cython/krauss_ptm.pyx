import cython
import numpy as np
cimport numpy as np
from scipy.linalg.blas import zgemm
from pauli cimport ComputeLexicographicPaulis

ctypedef double complex complex128_t

cdef extern from "math.h":
	double pow(double a, double x) nogil

cdef extern from "complex.h":
	double complex I
	double creal(double complex x) nogil
	double complex conj(double complex x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex128_t ZTrace(complex128_t [:, :] A, long dim):
	# Compute the trace of a matrix.
	cdef:
		int i
		complex128_t trace
	for i in range(dim):
		trace = trace + A[i, i]
	return trace

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ComputeDagger(complex128_t [:, :] A, complex128_t [:, :] Adag, long dim):
	# Compute the Hermitian conjugate of a matrix.
	cdef:
		int i, j
	for i in range(dim):
		for j in range(dim):
			Adag[i, j] = conj(A[j, i])
	return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def KrausToPTM(complex128_t [:, :, :] kraus, int nq):
	# Compute the PTM for a channel whose Kraus operators are given.
	# Given {K}, compute G given by G_ij = \sum_k Tr[ K_k P_i (K_k)^dag Pj ].
	cdef:
		int i, j, k
		long dim, n_pauli
	dim = long(pow(2, nq))
	n_pauli = long(pow(4, nq))

	# Compute Hermitian conjugate of Kraus operators
	cdef:
		complex128_t [:, :, :] krausH = np.zeros((n_pauli, dim, dim), dtype = np.complex128)
	for k in range(n_pauli):
		ComputeDagger(kraus[k, :, :], krausH[k, :, :], dim)

	# Preparing the Pauli operators.
	cdef:
		int [:, :] pauli_operators = np.zeros((n_pauli, nq), dtype = np.int32)
		complex128_t [:, :, :] pauli_matrices = np.zeros((n_pauli, dim, dim), dtype = np.complex128)
		int empty[0]

	ComputeLexicographicPaulis(nq, pauli_matrices, 0, empty, 1, pauli_operators)

	cdef:
		double [:, :] ptm = np.zeros((n_pauli, n_pauli), dtype = np.float64)
		complex128_t [:, :] prod = np.zeros((dim, dim), dtype = np.complex128)
		complex128_t alpha, beta
		double innerprod

	alpha = 1 + 0 * I
	beta = 0 + 0 * I
	# Compute the elements of PTM
	for i in range(n_pauli):
		for j in range(n_pauli):
			for k in range(kraus.shape[0]):
				zgemm(alpha, kraus[k, :, :], pauli_matrices[i, :, :], beta, prod, 0, 0, 1)
				zgemm(alpha, prod, krausH[k, :, :], beta, prod, 0, 0, 1)
				zgemm(alpha, prod, pauli_matrices[j, :, :], beta, prod, 0, 0, 1)
				innerprod = creal(ZTrace(prod, dim))
				ptm[i, j] = ptm[i, j] + innerprod/dim
			# print("PTM[%d, %d] was computed in %g seconds." % (i, j, timer() - click))
	ptm_tensor = ptm.reshape(tuple([4, 4]*nq))
	return ptm_tensor