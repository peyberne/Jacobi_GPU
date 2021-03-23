//#include <stdio.h>
//#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudalloc.h"
#include "unistd.h"



__device__ int xyindex(int i,  int nx )
{
   return i%nx;
}


__device__ int disp(int i, int j, int k, int nx, int ny)
{
   return i + j*nx + k*nx*ny;
}

__device__ void jacobiDevice(double *A, double *Anew, int nx, int ny)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int ind1=xyindex(index,nx);
   int d1=disp(1,0,0,nx,ny);
   int d2=disp(0,1,0,nx,ny);
   if (index<(nx*ny-d2) && index>d2 && ind1!=0 && ind1!=(nx-1) ){
      Anew[index] = A[index-d2] + A[index-d1] + A[index+d1] + A[index+d2];
      Anew[index] *= 0.25;
   }

}

__device__ void setValuesDevice(double *A, double *Anew, int nx, int ny)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int ind1=xyindex(index,nx);
   int d1=disp(1,0,0,nx,ny);
   int d2=disp(0,1,0,nx,ny);
   if (index<(nx*ny-d2) && index>d2 && ind1!=0 && ind1!=(nx-1) ){
      A[index] = Anew[index];
   }
}

__global__ void jacobi(double *A, double *Anew, int nx, int ny){
     jacobiDevice(A, Anew, nx, ny);
}

__global__ void setValues(double *A, double *Anew, int nx, int ny){
     setValuesDevice(A, Anew, nx, ny);
}

extern "C" void jacobi_cuda_(double *A, double *Anew,  int *nx, int *ny){
   int n=(*nx)*(*ny);
   int threadsPerBlock = 256;
   int numBlocks = ((n + threadsPerBlock -1) / threadsPerBlock);
   jacobi<<<numBlocks, threadsPerBlock>>>(A, Anew,  *nx, *ny);
//   cudaDeviceSynchronize();
   setValues<<<numBlocks, threadsPerBlock>>>(A, Anew,  *nx, *ny);
//   cudaDeviceSynchronize();
}

extern "C" void jacobi_cuda_opt_(double *A, double *Anew,  int *nx, int *ny){
   int n=(*nx)*(*ny);
   int threadsPerBlock = 256;
   int numBlocks = ((n + threadsPerBlock -1) / threadsPerBlock);
   jacobi<<<numBlocks, threadsPerBlock>>>(A, Anew,  *nx, *ny);
}

extern "C" void synchronize_cuda_device_(){
   cudaDeviceSynchronize();
}
