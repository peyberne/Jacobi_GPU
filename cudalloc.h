#include <stdio.h>
//#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct cuda_struct{
   double *coef_der_cuda;
   double *coef_int_cuda;
   double *coef_der1_stag;
   int *nz, *ny, *nx;
};
