#ifndef TRACEDOT_H
#define TRACEDOT_H

#include <complex.h>

typedef double complex complex128_t;

/*
	Compute the trace of the dot product of two matrices, i.e., Tr(A . B)
	The second parameter can be replaced by B^dag depending on the "dag" flag.
*/
extern complex128_t TraceDot(complex128_t **A, complex128_t **B, int rowsA, int colsA, int dag);

#endif /* TRACEDOT_H */