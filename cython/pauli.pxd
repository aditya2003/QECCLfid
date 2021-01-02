ctypedef double complex complex128_t

# Compute the matrix for a n-qubit Pauli operator.
cdef int ComputePauliMatrix(int [:] pauli_op, int nq, complex128_t [:, :] pauli_matrix)

# Compute all the n-qubit Pauli matrices in the lexicographic ordering.
cdef long ComputeLexicographicPaulis(int nq, complex128_t [:, :, :] pauli_matrices, int transpose, int [:] transpose_signs, int operators, int [:, :] pauli_operators)