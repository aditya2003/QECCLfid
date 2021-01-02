KrausPauliInnerProduct(kraus[k, :, :], pauli_operators[i, :], nq)
KrausPauliInnerProduct(kraus[k, :, :].T.conj(), pauli_operators[j, :])

					

def KrausPauliInnerProduct(int [:, :] kraus, int [:] pauli_op, int nq):
	# Compute the inner product between a Kraus operator and a Pauli matrix.
	# The Kraus operator is presented as a 2^n x 2^n complex matrix, while the Pauli operator is provided as an integer array.
	## Define Pauli matrices
	cdef:
		Py_ssize_t i, j, q
		complex[:, :, :] Paulis = np.zeros((4, 2, 2), dtype = complex)
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
	cdef:
		int [:] supp_K = np.arange(nq, dtype = int)

	# Define the n-qubit Pauli operator
	cdef:
		np.ndarray(complex, ndim=nq) pauli_operator = np.zeros((nq, 2, 2), dtype = complex)
	for q in range(nq):
		for i in range(2):
			for j in range(2):
				pauli_operator[q, i, j] = Paulis[pauli_op[q], i, j]


