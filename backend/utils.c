#include <stdio.h>
#include <complex.h>

/**
 * @brief Print a complex 2D array
 **/
void PrintArray2DComplexDouble(double complex **mat, char *name, long rows, long cols){
	printf("Array %s:\n", name);
	for (long r = 0; r < rows; r ++){
		for (long c = 0; c < cols; c ++){
			printf("   %g + i %g", creal(mat[r][c]), cimag(mat[r][c]));
		}
		printf("\n");
	}
	printf("======\n");
}