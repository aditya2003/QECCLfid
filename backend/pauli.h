#ifndef PAULI_H
#define PAULI_H

#include <complex.h>

typedef double complex complex128_t;

/*
	Compute the matrix for a n-qubit Pauli operator.
	Define Pauli matrices
*/
extern void ComputePauliMatrix(short *pauli_op, int nq, complex128_t **pauli_matrix);


// Compute all the n-qubit Pauli matrices in the lexicographic ordering.
extern void ComputeLexicographicPaulis(int nq, complex128_t ***pauli_matrices, short transpose, short *transpose_signs, short operators, short **pauli_operators);

#endif /* PAULI_H */