#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "pauli.h"

typedef double complex complex128_t;

void ComputePauliMatrix(short *pauli_op, int nq, complex128_t **pauli_matrix){
	// Compute the matrix for a n-qubit Pauli operator.
	// Define Pauli matrices
	int i, j, k;
	complex128_t Paulis[4][2][2];
	
	for (i = 0; i < 4; i ++)
		for (j = 0; j < 2; j ++)
			for (k = 0; k < 2; k ++)
				Paulis[i][j][k] = 0 + 0 * I;

	// Pauli matrix: I
	Paulis[0][0][0] = 1 + 0 * I;
	Paulis[0][1][1] = 1 + 0 * I;
	// Pauli matrix: X
	Paulis[1][0][1] = 1 + 0 * I;
	Paulis[1][1][0] = 1 + 0 * I;
	// Pauli matrix: Y
	Paulis[2][0][1] = 0 - 1 * I;
	Paulis[2][1][0] = 0 + 1 * I;
	// Pauli matrix: Z
	Paulis[3][0][0] = 1 + 0 * I;
	Paulis[3][1][1] = -1 + 0 * I;

	int b, num;
	long dim = (long) (pow(2, nq));

	short **binary = malloc(sizeof(short *) * dim);

	// Compute the binary representation of all numbers from 0 to 4^n-1.
	for (i = 0; i < dim; i ++){
		binary[i] = malloc(sizeof(short) * nq);
		num = i;
		for (b = 0; b < nq; b ++){
			binary[i][nq - b - 1] = num % 2;
			num = num / 2;
		}
	}

	// Compute the Pauli tensor products.
	int q;
	for (i = 0; i < dim; i ++){
		for (j = 0; j < dim; j ++){
			pauli_matrix[i][j] = 1 + 0 * I;
			for (q = 0; q < nq; q ++)
				pauli_matrix[i][j] = pauli_matrix[i][j] * Paulis[pauli_op[q]][binary[i][q]][binary[j][q]];
		}
	}

	// Free memory
	for (i = 0; i < dim; i ++)
		free(binary[i]);
	free(binary);
}


void ComputeLexicographicPaulis(int nq, complex128_t ***pauli_matrices, short transpose, short *transpose_signs, short operators, short **pauli_operators){
	// Compute all the n-qubit Pauli matrices in the lexicographic ordering.
	long n_pauli = (long)(pow(4, nq));
	short *pauli_op = malloc(sizeof(short) * nq);
	
	// Compute all the n-qubit Pauli matrices
	int i, q, index, y_count;
	for (i = 0; i < n_pauli; i ++){
		index = i;
		for (q = 0; q < nq; q ++){
			pauli_op[nq - q - 1] = index % 4;
			index = index / 4;
		}

		// Store the Pauli operator in an array if needed
		if (operators == 1)
			for (q = 0; q < nq; q ++)
				pauli_operators[i][q] = pauli_op[q];

		// Compute the Pauli matrix, which is a tensor product of the Pauli operators.
		ComputePauliMatrix(pauli_op, nq, pauli_matrices[i]);
		
		if (transpose == 1){
			// Count the number of Ys' for computing the transpose.
			y_count = 0;
			for (q = 0; q < nq; q ++)
				if (pauli_op[q] == 2)
					y_count += 1;
			transpose_signs[i] = (short) pow(-1, y_count);
		}
	}

	// Free memory
	free(pauli_op);

	/*
	// Print the Pauli operators.
	int j, k;
	long dim = (long) pow(2, nq);
	printf("%ld Pauli operators\n", n_pauli);
	for (k = 0; k < n_pauli; k ++){
		printf("P_%d\n", k);
		for (i = 0; i < dim; i ++){
			for (j = 0; j < dim; j ++){
				printf("%d + i %d   ", (int) creal(pauli_matrices[k][i][j]), (int) cimag(pauli_matrices[k][i][j]));
			}
			printf("\n");
		}
		printf("------\n");
	}
	*/
}