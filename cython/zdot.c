#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#include "zdot.h"


void ZDot(double complex **matA, double complex **matB, double complex **prod, int rowsA, int colsA, int rowsB, int colsB){
	/*
		Multiply two complex matrices.
		For high-performance, we will use the zgemm function of the BLAS library.
		See https://software.intel.com/en-us/node/520775 .
		The zgemm function is defined with the following parameters.
		extern cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						   int m, // number of rows of A
						   int n, // number of columns of B
						   int k, // number of columns of A = number of rows of B
						   double alpha, // scalar factor to be multiplied to the product.
						   double complex *A, // input matrix A
						   int k, // leading dimension of A
						   double complex *B, // input matrix B
						   int n, // leading dimension of B
						   double beta, // relative shift from the product, by a factor of C.
						   double complex *C, // product of A and B
						   int n //leading dimension of C.
						  );
	*/
	if (colsA != rowsB)
		printf("Cannot multiply matrices of shape (%d x %d) and (%d x %d).\n", rowsA, colsA, rowsB, colsB);
	else{
		MKL_INT m = rowsA, n = colsB, k = colsA;
		MKL_Complex16 A[rowsA * colsA], B[rowsB * colsB], C[rowsA * colsB], alpha, beta;
		int i, j;
		for (i = 0; i < rowsA; i ++){
			for (j = 0; j < colsA; j ++){
				A[i * colsA + j].real = creal(matA[i][j]);
				A[i * colsA + j].imag = cimag(matA[i][j]);
			}
		}
		for (i = 0; i < rowsB; i ++){
			for (j = 0; j < colsB; j ++){
				B[i * colsB + j].real = creal(matB[i][j]);
				B[i * colsB + j].imag = cimag(matB[i][j]);
			}
		}
		alpha.real = 1;
		alpha.imag = 0;
		beta.real = 0;
		beta.imag = 0;
		// Call the BLAS function.
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C, n);
		// Load the product
		for (i = 0; i < rowsA; i ++)
			for (j = 0; j < colsB; j ++)
				prod[i][j] = C[i * colsB + j].real + C[i * colsB + j].imag * I;
		/*
		// Print the result
		printf("Product\n");
		for (i = 0; i < rowsA; i ++){
			for (j = 0; j < colsB; j ++){
				printf("%.3f + i %.3f  ", creal(prod[i][j]), cimag(prod[i][j]));
			}
			printf("\n");
		}
		printf("------\n");
		*/
	}
}