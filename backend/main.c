#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "krauss_ptm.h"
#include "krauss_theta.h"

typedef double complex complex128_t;

int main(int argc, char **argv){
	/*
	This function is simply to test all the C functions in the converted/ folder.
	*/
	clock_t begin, end;
	double runtime;
	printf("convert: source last compiled on %s at %s.\n", __DATE__, __TIME__);
	if (argc < 2){
		printf("Usage: ./convert <nq>\n");
		return 0;
	}

	int i, j, k, nq;
	sscanf(argv[1], "%d", &nq);
	
	long n_pauli = (long) pow(4, nq), dim = (long) pow(2, nq);
	double *kraus_real = malloc(sizeof(double) * (n_pauli * dim * dim));
	double *kraus_imag = malloc(sizeof(double) * (n_pauli * dim * dim));
	
	for (k = 0; k < n_pauli; k ++){
		for (i = 0; i < dim; i ++){
			for (j = 0; j < dim; j ++){
				kraus_real[k * dim * dim + i * dim + j] = ((double) rand()) / ((double) ((unsigned) RAND_MAX + 1));
				kraus_imag[k * dim * dim + i * dim + j] = ((double) rand()) / ((double) ((unsigned) RAND_MAX + 1));
			}
		}
	}

	// Print the Kraus operators.
	printf("Kraus operators.\n");
	for (k = 0; k < n_pauli; k ++){
		printf("K_%d\n", k);
		for (i = 0; i < dim; i ++){
			for (j = 0; j < dim; j ++){
				printf("%.3f + i %.3f   ", kraus_real[k * dim * dim + i * dim + j], kraus_imag[k * dim * dim + i * dim + j]);
			}
			printf("\n");
		}
		printf("-------\n");
	}
	printf("=======\n");

	// Compute Theta.
	long size = n_pauli * n_pauli;
	begin = clock();
	double *theta_flat = KrausToTheta(kraus_real, kraus_imag, nq, -1);
	double real_part, imag_part;
	// Print the Theta matrix.
	printf("=======\n");
	printf("Theta\n");
	for (i = 0; i < n_pauli; i ++){
		for (j = 0; j < n_pauli; j ++){
			real_part = theta_flat[i * n_pauli + j];
			imag_part = theta_flat[size + i * n_pauli + j];
			printf("%.3f + i %.3f   ", real_part, imag_part);
		}
		printf("\n");
	}
	printf("=======\n");

	// Estimate the time for computing Theta.
	end = clock();
	runtime = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("***********\n");
	printf("Theta was computed in %d seconds.\n", (int) runtime);
	printf("***********\n");

	// Compute PTM.
	begin = clock();
	double *ptm_flat = KrausToPTM(kraus_real, kraus_imag, nq, -1);
	// Print the PTM.
	printf("=======\n");
	printf("PTM\n");
	for (i = 0; i < n_pauli; i ++){
		for (j = 0; j < n_pauli; j ++){
			printf("%.3f  ", ptm_flat[i * n_pauli + j]);
		}
		printf("\n");
	}
	printf("=======\n");
	
	// Estimate the time for computing PTM.
	end = clock();
	runtime = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("***********\n");
	printf("PTM was computed in %d seconds.\n", (int) runtime);
	printf("***********\n");

	// Free memory.
	free(kraus_real);
	free(kraus_imag);
	
	return 0;
}