/*#include <stdio.h>
//#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
struct cuda_struct{
   double *coef_der_cuda;
};

*/
#include "cudalloc.h"
/*__global__ void set_coefficients(cuda_struct *structure){
   structure->coef_der_cuda[0] = 1.0/12.0;
   structure->coef_der_cuda[1] = -2.0/3.0;
   structure->coef_der_cuda[2] = 2.0/3.0;
   structure->coef_der_cuda[3] = -1.0/12.0;
*/
__global__ void set_coefficients(double *coef, double *coefstag, double *coef_int){
coef[0] = 1.0/12.0;
coef[1] = -2.0/3.0;
coef[2] = 2.0/3.0;
coef[3] = -1.0/12.0;
coef_int[0] = -1.0/16.0;
coef_int[1] = 9.0/16.0;
coef_int[2] = 9.0/16.0;
coef_int[3] = -1.0/16.0;
coefstag[0] = 1.0/24.0;
coefstag[1] = -9.0/8.0;
coefstag[2] = 9.0/8.0;
coefstag[3] = -1.0/24.0;
}
void set_extent(int *out, int in){
*out = in;
}
extern "C" void allocate_cuda_struct(cuda_struct *& structure){
   structure = new cuda_struct();
   return;
}

//__global__ void print_coefficients(cuda_struct *structure){
__global__ void print_coefficients(double *coef){
}
extern "C" void fill_cuda_struct(cuda_struct *structure, int *izeg, int *izsg, int *iyeg, int *iysg, int *ixeg, int *ixsg){
//extern "C" void fill_cuda_struct(cuda_struct *structure){

   cudaMallocManaged( (void **)&(structure->coef_der_cuda), sizeof(double) * 4 );
   cudaMallocManaged( (void **)&(structure->coef_der1_stag), sizeof(double) * 4 );
   cudaMallocManaged( (void **)&(structure->coef_int_cuda), sizeof(double) * 4 );
   set_coefficients<<<1, 1>>>(structure->coef_der_cuda, structure->coef_der1_stag, structure->coef_int_cuda);
   //set_coefficients<<<1, 1>>>(structure->coef_der_cuda);
   //set_extent((structure->nz), (*izeg-*izsg+1));
   //set_extent(&(structure->ny), (*izeg-*izsg+1));
   //set_extent(&(structure->nx), (*izeg-*izsg+1));
   //set_extent(&(structure->nx), (*iyeg-*iysg+1));
   //&(structure->nz) = &k;//(*izeg-*izsg+1);
   //&(structure->ny) = &k;//(*ixeg-*ixsg+1);
   //&(structure->nx) = &k;//(*iyeg-*iysg+1);
   //printf("AAAAAAA %d %d %d\n", structure->nz, structure->ny, structure->nx);
   //set_extent<<<1, 1>>>(structure->nz, (*izeg-*izsg+1));
   //set_extent<<<1, 1>>>(structure->ny, (*ixeg-*ixsg+1));
   //set_extent<<<1, 1>>>(structure->nx, (*iyeg-*iysg+1));
   cudaDeviceSynchronize();

   printf("cudalloc coef_int %f %f %f %f \n", structure->coef_int_cuda[0], structure->coef_int_cuda[1], structure->coef_int_cuda[2], structure->coef_int_cuda[3]);
   return;
}

extern "C" void *allocate_cuda_memory(int *n){
   double *a;
   int N=*n;
   //a = (float *)malloc(*n*sizeof(float));
   cudaMallocManaged( (void **)&a, sizeof(double) * N );
   cudaDeviceSynchronize();
   return (void *) a;
}



