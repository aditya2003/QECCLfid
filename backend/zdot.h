#ifndef ZDOT_H
#define ZDOT_H

typedef double complex complex128_t;

#include <complex.h>

/*
	Multiply two complex matrices A and B to produce a matrix C := A . B.
	We will use the BLAS routine zgemm to multiply matrices. See https://software.intel.com/en-us/node/520775.
	Inputs:
		double complex **A: complex matrix of shape (i x k)
		double complex **B: complex matrix of shape (k x j)
		double complex **C: complex matrix of shape (i x j)
		int i: number of rows in A
		int k: number of columns in A (also, number of rows in B)
		int j: number of columns in B
		In the function definition, we use
			A --> matA
			B --> matB
			C --> prod
			i --> rowsA
			k --> colsA ,which is also rowsB
			j --> colsB
*/
extern void ZDot(complex128_t **matA, complex128_t **matB, complex128_t **prod, int rowsA, int colsA, int rowsB, int colsB);

#endif /* ZDOT_H */