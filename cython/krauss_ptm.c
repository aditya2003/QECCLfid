#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "zdot.h"
#include "pauli.h"
#include "tracedot.h"
#include "krauss_ptm.h"


typedef double complex complex128_t;

complex128_t ZTrace(complex128_t **A, long dim){
	// Compute the trace of a matrix.
	int i;
	complex128_t trace = 0;
	for (i = 0; i < dim; i ++)
		trace += A[i][i];
	// printf("Trace = %.3f + i %.3f.\n", creal(trace), cimag(trace));
	return trace;
}

void ComputeDagger(complex128_t **A, complex128_t **Adag, long dim){
	// Compute the Hermitian conjugate of a matrix.
	int i, j;
	for (i = 0; i < dim; i ++)
		for (j = 0; j < dim; j ++)
			Adag[i][j] = conj(A[j][i]);
}


double* KrausToPTM(double *kraus_real, double *kraus_imag, int nq){
	/*
		Compute the PTM for a channel whose Kraus operators are given.
		Given {K}, compute G given by G_ij = \sum_k Tr[ K_k P_i (K_k)^dag Pj ].
		
		To do:
		double [:, :] ptm = np.zeros((n_pauli, n_pauli), dtype = np.float64)
	*/
	int i, j, k;
	long dim = (long)(pow(2, nq)), n_pauli = (long)(pow(4, nq));
	
	// Compute Hermitian conjugate of Kraus operators
	complex128_t ***kraus = malloc(sizeof(complex128_t **) * n_pauli);
	complex128_t ***krausH = malloc(sizeof(complex128_t **) * n_pauli);
	double real_part, imag_part;

	for (k = 0; k < n_pauli; k ++){
		kraus[k] = malloc(sizeof(complex128_t *) * dim);
		krausH[k] = malloc(sizeof(complex128_t *) * dim);
		for (i = 0; i < dim; i ++){
			kraus[k][i] = malloc(sizeof(complex128_t) * dim);
			krausH[k][i] = malloc(sizeof(complex128_t) * dim);
			for (j = 0; j < dim; j ++){
				real_part = kraus_real[k * dim * dim + i * dim + j];
				imag_part = kraus_imag[k * dim * dim + i * dim + j];
				kraus[k][i][j] = real_part + I * imag_part;
			}
		}
		ComputeDagger(kraus[k], krausH[k], dim);
	}
	
	// Preparing the Pauli operators.
	short **pauli_operators = malloc(sizeof(short *) * n_pauli);
	complex128_t ***pauli_matrices = malloc(sizeof(complex128_t **) * n_pauli);
	for (i = 0; i < n_pauli; i ++){
		pauli_operators[i] = malloc(sizeof(short) * nq);
		pauli_matrices[i] = malloc(sizeof(complex128_t *) * dim);
		for (j = 0; j < dim; j ++){
			pauli_matrices[i][j] = malloc(sizeof(complex128_t) * dim);
		}
	}

	ComputeLexicographicPaulis(nq, pauli_matrices, 0, NULL, 1, pauli_operators);
	
	// Compute the elements of PTM.
	double **ptm = malloc(sizeof(double *) * n_pauli);
	
	complex128_t **prod = malloc(sizeof(complex128_t *) * dim);
	for (i = 0; i < dim; i ++)
		prod[i] = malloc(sizeof(complex128_t) * dim);

	for (i = 0; i < n_pauli; i ++){
		ptm[i] = malloc(sizeof(double) * n_pauli);
		for (j = 0; j < n_pauli; j ++){
			ptm[i][j] = 0;
			for (k = 0; k < n_pauli; k ++){
				ZDot(kraus[k], pauli_matrices[i], prod, dim, dim, dim, dim);
				ZDot(prod, krausH[k], prod, dim, dim, dim, dim);
				// ZDot(prod, pauli_matrices[j], prod, dim, dim, dim, dim);
				// ptm[i][j] += creal(ZTrace(prod, dim));
				ptm[i][j] += creal(TraceDot(prod, pauli_matrices[j], dim, dim, 0));
				// printf("PTM_%d[%d, %d] = %.3f\n", k, i, j, ptm[i][j]);
			}
			ptm[i][j] = ptm[i][j]/((double) dim);
			// printf("PTM[%d, %d] = %.3f\n", i, j, ptm[i][j]);
		}
	}

	// Flatten the PTM array to return it.
	double *ptm_flat = malloc(sizeof(double) * (n_pauli * n_pauli));
	for (i = 0; i < n_pauli; i ++)
		for (j = 0; j < n_pauli; j ++)
			ptm_flat[i * n_pauli + j] = ptm[i][j];

	// Free memory

	for (i = 0; i < dim; i ++)
		free(prod[i]);
	free(prod);

	for (k = 0; k < n_pauli; k ++){
		for (i = 0; i < dim; i ++){
			free(kraus[k][i]);
			free(krausH[k][i]);
			free(pauli_matrices[k][i]);
		}
		free(kraus[k]);
		free(krausH[k]);
		free(pauli_operators[k]);
		free(pauli_matrices[k]);
		free(ptm[k]);
	}
	free(kraus);
	free(krausH);
	free(pauli_operators);
	free(pauli_matrices);
	free(ptm);

	return ptm_flat;
}