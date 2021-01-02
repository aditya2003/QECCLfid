import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
	double pow(double a, double x) nogil

cdef extern from "complex.h":
	double complex I

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ComputePauliMatrix(int [:] pauli_op, int nq, complex128_t [:, :] pauli_matrix):
	# Compute the matrix for a n-qubit Pauli operator.
	## Define Pauli matrices
	cdef:
		int i, j, k
		complex128_t Paulis[4][2][2]
	
	for i in range(4):
		for j in range(2):
			for k in range(2):
				Paulis[i][j][k] = 0

	# Pauli matrix: I
	Paulis[0][0][0] = 1
	Paulis[0][1][1] = 1
	# Pauli matrix: X
	Paulis[1][0][1] = 1
	Paulis[1][1][0] = 1
	# Pauli matrix: Y
	Paulis[1][0][1] = I
	Paulis[1][1][0] = -I
	# Pauli matrix: Z
	Paulis[1][0][0] = 1
	Paulis[1][1][1] = -1

	cdef:
		int b, num
		long dim

	dim = int(pow(2, nq))

	cdef:
		int [:, :] binary = np.zeros((dim, nq), dtype = np.int32)
	# Compute the binary representation of all numbers from 0 to 4^n-1.
	for i in range(dim):
		num = i
		for b in range(nq):
			binary[i, nq - b - 1] = num % 2
			num = num // 2

	# Compute the Pauli tensor products.
	cdef:
		int q
	for i in range(dim):
		for j in range(dim):
			pauli_matrix[i, j] = 1 + 0 * I
			for q in range(nq):
				pauli_matrix[i, j] = pauli_matrix[i, j] * Paulis[pauli_op[q]][binary[i, q]][binary[j, q]]
	return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long ComputeLexicographicPaulis(int nq, complex128_t [:, :, :] pauli_matrices, int transpose, int [:] transpose_signs, int operators, int [:, :] pauli_operators):
	# Compute all the n-qubit Pauli matrices in the lexicographic ordering.
	cdef:
		long dim, n_pauli
		int [:] pauli_op = np.zeros(nq, dtype = np.int32)
	dim = long(pow(2, nq))
	n_pauli = long(pow(4, nq))

	# Compute all the n-qubit Pauli matrices
	cdef:
		int i, q, index, y_count
	for i in range(n_pauli):
		index = i
		for q in range(nq):
			pauli_op[nq - q - 1] = index % 4
			index = index // 4

		# Store the Pauli operator in an array if needed
		if operators == 1:
			for q in range(nq):
				pauli_operators[i, q] = pauli_op[q]

		# Compute the Pauli matrix, which is a tensor product of the Pauli operators.
		ComputePauliMatrix(pauli_op, nq, pauli_matrices[i, :, :])
		
		if transpose == 1:
			# Count the number of Ys' for computing the transpose.
			y_count = 0
			for q in range(nq):
				if (pauli_op[q] == 2):
					y_count += 1
			transpose_signs[i] = (-1) ** y_count
	
	return n_pauli