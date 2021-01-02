cimport cython

# ctypedef double complex complex128_t

cdef extern from "complex.h":
	double complex conj(double complex x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex128_t TraceDot(complex128_t [:, :] A, complex128_t [:, :] B, int rowsA, int colsA, int dag):
	# Compute the trace of the dot product of two matrices, i.e., Tr(A . B)
	# The second parameter can be replaced by B^dag depending on the "dag" flag.
	cdef:
		Py_ssize_t i, j
		complex128_t result
	result = 0
	if dag == 0:
		for i in range(rowsA):
			for j in range(colsA):
				result = result + A[i, j] * B[j, i]
	else:
		for i in range(rowsA):
			for j in range(colsA):
				result = result + A[i, j] * conj(B[i, j])
	return result