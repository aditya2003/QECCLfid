#ifndef KRAUSS_THETA_H
#define KRAUSS_THETA_H

#include <complex.h>

typedef double complex complex128_t;

/*
	Convert from the Kraus representation to the "Theta" representation.
	The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
	X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
		   = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
	So we find that
	T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	
	To do:
	complex128_t [:, :] theta = np.zeros((n_pauli, n_pauli), dtype = np.complex128)
*/
extern double* KrausToTheta(double *kraus_real, double *kraus_imag, int nq);

#endif /* KRAUSS_THETA_H */