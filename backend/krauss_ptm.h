#ifndef KRAUSS_PTM_H
#define KRAUSS_PTM_H

#include <complex.h>

typedef double complex complex128_t;

/*
	Compute the PTM for a channel whose Kraus operators are given.
	Given {K}, compute G given by G_ij = \sum_k Tr[ K_k P_i (K_k)^dag Pj ].
*/
extern double* KrausToPTM(double *kraus_real, double *kraus_imag, int nq, long nkr);

#endif /* KRAUSS_PTM_H */