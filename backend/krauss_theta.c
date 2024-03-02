#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "utils.h"
#include "pauli.h"
#include "tracedot.h"
#include "krauss_theta.h"

typedef double complex complex128_t;

void MultiplyScalar(complex128_t scalar, complex128_t **matrix, int rows, int cols){
	// Multiply the entries of a matrix with a scalar.
	int i, j;
	for (i = 0; i < rows; i ++)
		for (j = 0; j < cols; j ++)
			matrix[i][j] = matrix[i][j] * scalar;
}


void Add(complex128_t **A, complex128_t **B, long rows, long cols){
	// Add two matrices and overwrite the second matrix with the sum.
	int i, j;
	for (i = 0; i < rows; i ++)
		for (j = 0; j < cols; j ++)
			B[i][j] = B[i][j] + A[i][j];
}


double* KrausToTheta(double *kraus_real, double *kraus_imag, int nq, long nkr){
	/*
		Convert from the Kraus representation to the "Theta" representation.
		The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
		T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
		Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
		X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
			 = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
		So we find that
		T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	*/

	int i, j, k, q;
	long dim = (long)(pow(2, nq)), n_pauli = (long)(pow(4, nq));
	double real_part, imag_part;

	// If the number of Kraus operators is 4^n, it can be left in the input as -1.
	if (nkr == -1)
		nkr = n_pauli;

	complex128_t ***kraus = malloc(sizeof(complex128_t **) * nkr);
	for (k = 0; k < nkr; k ++){
		kraus[k] = malloc(sizeof(complex128_t *) * dim);
		// printf("K_%d\n", k);
		for (i = 0; i < dim; i ++){
			kraus[k][i] = malloc(sizeof(complex128_t) * dim);
			for (j = 0; j < dim; j ++){
				real_part = kraus_real[k * dim * dim + i * dim + j];
				imag_part = kraus_imag[k * dim * dim + i * dim + j];
				kraus[k][i][j] = real_part + I * imag_part;
				// printf("%g + i %g  ", real_part, imag_part);
			}
			// printf("\n");
		}
		// printf("------\n");
	}

	// printf("n_pauli s= %ld and dim = %ld\n", n_pauli, dim);
	
	// Allocate memory for the Pauli operators.
	short *transpose_signs = malloc(sizeof(short) * n_pauli);
	short **pauli_operators = malloc(sizeof(int *) * n_pauli);
	complex128_t ***pauli_matrices = malloc(sizeof(complex128_t **) * n_pauli);
	
	for (i = 0; i < n_pauli; i ++){
		pauli_operators[i] = malloc(sizeof(short) * nq);
		pauli_matrices[i] = malloc(sizeof(complex128_t *) * dim);
		for (j = 0; j < dim; j ++)
			pauli_matrices[i][j] = malloc(sizeof(complex128_t) * dim);
	}
	// Compute Pauli operators
	ComputeLexicographicPaulis(nq, pauli_matrices, 1, transpose_signs, 1, pauli_operators);

	// printf("Computed Pauli operators\n.");

	// Allocate memory for the Chi and Theta matrices
	complex128_t tr_Pi_K, tr_Pj_Kdag, transpose_sign;
	short *pauli_op_2nq = malloc(sizeof(short) * (2 * nq));
	complex128_t **pauli_matrix_2nq = malloc(sizeof(complex128_t *) * n_pauli);
	complex128_t **chi = malloc(sizeof(complex128_t *) * n_pauli);
	complex128_t **theta = malloc(sizeof(complex128_t *) * n_pauli);
	for (i = 0; i < n_pauli; i ++){
		chi[i] = malloc(sizeof(complex128_t) * n_pauli);
		theta[i] = malloc(sizeof(complex128_t) * n_pauli);
		pauli_matrix_2nq[i] = malloc(sizeof(complex128_t) * n_pauli);
		for (j = 0; j < n_pauli; j ++){
			chi[i][j] = 0 + 0 * I;
			theta[i][j] = 0 + 0 * I;
			pauli_matrix_2nq[i][j] = 0 + 0 * I;
		}
	}

	// Compute the Chi and Theta matrix.
	for (i = 0; i < n_pauli; i ++){
		for (j = 0; j < n_pauli; j ++){
			chi[i][j] = 0 + 0 * I;
			// Compute the Chi matrix first.
			// printf("Computing Chi[%d, %d]\n", i, j);
			// Since Chi is Hermitian, we only need to really compute its upper diagonal.
			if (i <= j){
				for (k = 0; k < nkr; k ++){
					tr_Pi_K = TraceDot(pauli_matrices[i], kraus[k], dim, dim, 0);
					tr_Pj_Kdag = TraceDot(pauli_matrices[j], kraus[k], dim, dim, 1);
					chi[i][j] += tr_Pi_K * tr_Pj_Kdag;
				}
				chi[i][j] = chi[i][j]/((double) (n_pauli));
			}
			else
				chi[i][j] = conj(chi[j][i]);
			
			// Consistency check: the diagonal elements should be posivite real numbers.
			if (i == j){
				if ((creal(chi[i][j]) <= -1E-14) || (fabs(cimag(chi[i][j])) >= 1E-14)){
					printf("Error: Chi[%d, %d] = %.3e + i %.3e\n", i, j, creal(chi[i][j]), cimag(chi[i][j]));
					exit(0);
				}
			}

			// printf("Chi[%d, %d] = %g + i %g.\n", i, j, creal(chi[i][j]), cimag(chi[i][j]));
			
			// Compute the tensor product of two Pauli operators.
			transpose_sign = transpose_signs[j] + I * 0;
			for (q = 0; q < nq; q ++)
				pauli_op_2nq[q] = pauli_operators[i][q];
			for (q = nq; q < 2 * nq; q ++)
				pauli_op_2nq[q] = pauli_operators[j][q - nq];
			ComputePauliMatrix(pauli_op_2nq, 2 * nq, pauli_matrix_2nq);
			MultiplyScalar(chi[i][j] * transpose_sign, pauli_matrix_2nq, n_pauli, n_pauli);
			Add(pauli_matrix_2nq, theta, n_pauli, n_pauli);
			
			// printf("Chi[%d, %d] = %g + i %g.\n", i, j, creal(chi[i][j]), cimag(chi[i][j]));
		}
		// printf("----\n");
	}

	// PrintArray2DComplexDouble(theta, "Theta", n_pauli, n_pauli);

	// Flatten the theta matrix to return it.
	double *theta_flat = malloc(sizeof(double) * (2 * n_pauli * n_pauli));
	long size = n_pauli * n_pauli;
	for (i = 0; i < n_pauli; i ++){
		for (j = 0; j < n_pauli; j ++){
			theta_flat[i * n_pauli + j] = creal(theta[i][j]);
			theta_flat[size + i * n_pauli + j] = cimag(theta[i][j]);
		}
	}

	/*
	printf("%ld Real entries\n", n_pauli * n_pauli);
	for (i = 0; i < n_pauli * n_pauli; i ++)
		printf("%.3f  ", theta_flat[i]);
	printf("\n");
	printf("%ld Complex entries\n", n_pauli * n_pauli);
	for (i = n_pauli * n_pauli; i < 2 * n_pauli * n_pauli; i ++)
		printf("%.3f  ", theta_flat[i]);
	printf("\n");
	*/

	// Free memory for the Kraus operators
	for (k = 0; k < nkr; k ++){
		for (i = 0; i < dim; i ++)
			free(kraus[k][i]);
		free(kraus[k]);
	}
	free(kraus);

	// Free memory allocated for Pauli matrices
	free(transpose_signs);
	for (i = 0; i < n_pauli; i ++){
		free(pauli_operators[i]);
		for (j = 0; j < dim; j ++)
			free(pauli_matrices[i][j]);
		free(pauli_matrices[i]);
	}
	free(pauli_matrices);

	// Free memory allocated for Chi matrix.
	free(pauli_op_2nq);
	for (i = 0; i < n_pauli; i ++){
		free(chi[i]);
		free(theta[i]);
		free(pauli_matrix_2nq[i]);
	}
	free(chi);
	free(theta);
	free(pauli_matrix_2nq);

	return theta_flat;
}