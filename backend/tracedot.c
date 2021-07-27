#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

typedef double complex complex128_t;

complex128_t TraceDot(complex128_t **A, complex128_t **B, int rowsA, int colsA, int dag){
	/*
		Compute the trace of the dot product of two matrices, i.e., Tr(A . B)
		The second parameter can be replaced by B^dag depending on the "dag" flag.
	*/
	int i, j;
	complex128_t result = 0;

	// printf("Function: TraceDot on matrix with %d rows and columns.\n", rowsA);
	
	if (dag == 0)
		for (i = 0; i < rowsA; i ++)
			for (j = 0; j < colsA; j ++)
				result += A[i][j] * B[j][i];
	
	else
		for (i = 0; i < rowsA; i ++)
			for (j = 0; j < colsA; j ++)
				result += A[i][j] * conj(B[i][j]);

	// printf("result = %.3f + %.3f\n", creal(result), cimag(result));

	return result;
}