//#include <stdio.h>
//#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudalloc.h"
#include "unistd.h"


__constant__ __device__ double coef_int[4] = {-1.0/16.0, 9.0/16.0, 9.0/16.0, -1.0/16.0 };
__constant__ __device__ double coef_der1_stag[4] = { +1.0/24.0, -9.0/8.0, 9.0/8.0, -1.0/24.0 };
__constant__ __device__ double coef_der2[5] =  {-1.0/12.0, 4.0/3.0, -5.0/2.0, +4.0/3.0, -1.0/12.0};
__constant__ __device__ double coef_der1_n2n[4] =  {  1.0/12.0,  -2.0/3.0,  2.0/3.0,  -1.0/12.0};
__constant__ __device__ double coef_der2_stag[4] = { 1.0/2., -1.0/2.0, -1.0/2., 1.0/2.0};

__device__ int xyindex(int i,  int nz )
{
   return i%nz;
}


__device__ int disp(int i, int j, int k, int nx, int ny)
{
   return i + j*nx + k*nx*ny;
}
/*
namespace detail {
__device__ void grady(double *f, double *fy, int nz, int ny, int nx, double fact)
{

   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d1=disp(1,0,0,nx,ny);
   int d2=disp(2,0,0,nx,ny);
   if (index<(nx*ny*nz-d2) && index>d2){
      fy[index] = coef_der1_n2n[0]*f[index-d2] + coef_der1_n2n[1]*f[index-d1] + coef_der1_n2n[2]*f[index+d1] + coef_der1_n2n[3]*f[index+d2];
      fy[index] *= fact;
   }
}

} // namespace detail

__global__ void grady(double *f, double *fy, int nz, int ny, int nx, double fact) { 
   detail::grady(f, fy,  nz, ny, nx, fact);
}
*/

__device__ void gradzDevice(double *f, double *fz, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d1=disp(0,0,1,nx,ny);
   int d2=disp(0,0,2,nx,ny);
   if (index<(nx*ny*nz-d2) && index>=d2){
      fz[index] = coef_der1_n2n[0]*f[index-d2] + coef_der1_n2n[1]*f[index-d1] + coef_der1_n2n[2]*f[index+d1] + coef_der1_n2n[3]*f[index+d2];
      fz[index] *= fact;
   }

}


__global__ void gradz(double *f, double *fz, int nz, int ny, int nx, double fact){
     gradzDevice(f, fz, nz, ny, nx, fact);
}

extern "C" void gradz_n2n_fd4_cuda_(double *f, double *f_z,  int *nz, int *ny, int *nx, double *fact){
   int n=(*nx)*(*ny)*(*nz);
   int threadsPerBlock = 256;
   int numBlocks = ((n + threadsPerBlock -1) / threadsPerBlock);
   gradz<<<numBlocks, threadsPerBlock>>>(f, f_z,  *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}
__device__ void gradyDevice(double *f, double *fy, int nz, int ny, int nx, double fact)
{

   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d1=disp(1,0,0,nx,ny);
   int d2=disp(2,0,0,nx,ny);
   if (index<(nx*ny*nz-d2) && index>d2){
      fy[index] = coef_der1_n2n[0]*f[index-d2] + coef_der1_n2n[1]*f[index-d1] + coef_der1_n2n[2]*f[index+d1] + coef_der1_n2n[3]*f[index+d2];
      fy[index] *= fact;
   }

}

__global__ void grady(double *f, double *fy, int nz, int ny, int nx, double fact)
{
   gradyDevice(f,fy, nz, ny, nx, fact);
}

extern "C" void grady_n2n_fd4_cuda_(double *f, double *f_y,   int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   grady<<<numBlocks, threadsPerBlock>>>(f, f_y,  *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}


__device__ void gradx_n2n_fd4Device(double *f, double *fx, int nz, int ny, int nx, double fact)
{
   int d1=disp(0,1,0,nx,ny);
   int d2=disp(0,2,0,nx,ny);
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   if (index<(nx*ny*nz-d2) && index>d2){
      fx[index] = coef_der1_n2n[0]*f[index-d2] + coef_der1_n2n[1]*f[index-d1] + coef_der1_n2n[2]*f[index+d1] + coef_der1_n2n[3]*f[index+d2];
      fx[index] *= fact;
   }

}
__global__ void gradx(double *f, double *fx, int nz, int ny, int nx, double fact)
{
   gradx_n2n_fd4Device(f,fx, nz, ny, nx, fact);
}
extern "C" void gradx_n2n_fd4_cuda_(double *f, double *f_x,   int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradx<<<numBlocks, threadsPerBlock>>>(f, f_x,  *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}
__device__ void gradz_v2n_fd4Device(double *f, double *f_z_v2n, int nz, int ny, int nx, double fact)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-1,0,-1,nx,ny);
   int d12=index+disp(-1,0,0,nx,ny);
   int d13=index+disp(-1,0,1,nx,ny);
   int d14=index+disp(-1,0,2,nx,ny);
   int d21=index+disp(0,0,-1,nx,ny);
   int d22=index+disp(0,0,0,nx,ny);
   int d23=index+disp(0,0,1,nx,ny);
   int d24=index+disp(0,0,2,nx,ny);
   int d31=index+disp(1,0,-1,nx,ny);
   int d32=index+disp(1,0,0,nx,ny);
   int d33=index+disp(1,0,1,nx,ny);
   int d34=index+disp(1,0,2,nx,ny);
   int d41=index+disp(2,0,-1,nx,ny);
   int d42=index+disp(2,0,0,nx,ny);
   int d43=index+disp(2,0,1,nx,ny);
   int d44=index+disp(2,0,2,nx,ny);

   if (d11>0 && d44<(nx*ny*nz)){
      f_z_v2n[index] = coef_int[0]  * (coef_der1_stag[0]*f[d11] + coef_der1_stag[1]*f[d12] +
                     coef_der1_stag[2]*f[d13] + coef_der1_stag[3]*f[d14]) + 
                     coef_int[1] * (coef_der1_stag[0]*f[d21] + coef_der1_stag[1]*f[d22] +
                     coef_der1_stag[2]*f[d23] + coef_der1_stag[3]*f[d24])+ 
                     coef_int[2]  * (coef_der1_stag[0]*f[d31] + coef_der1_stag[1]*f[d32] +
                     coef_der1_stag[2]*f[d33] + coef_der1_stag[3]*f[d34]) +
                     coef_int[3] *  (coef_der1_stag[0]*f[d41] + coef_der1_stag[1]*f[d42] +
                     coef_der1_stag[2]*f[d43] + coef_der1_stag[3]*f[d44]);
      f_z_v2n[index] *= fact;
   }

}

__global__ void gradz_v2n_fd4(double *f, double *f_z_v2n, int nz, int ny, int nx, double fact){
    gradz_v2n_fd4Device(f, f_z_v2n, nz, ny, nx, fact);
}
extern "C" void gradz_v2n_fd4_cuda_(double *f, double *f_z_v2n,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradz_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_z_v2n, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}


__device__ void grady_v2n_fd4Device(double *f, double *f_y_v2n, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-1,0,-1,nx,ny);
   int d12=index+disp(0,0,-1,nx,ny);
   int d13=index+disp(1,0,-1,nx,ny);
   int d14=index+disp(2,0,-1,nx,ny);
   int d21=index+disp(-1,0,0,nx,ny);
   int d22=index+disp(0,0,0,nx,ny);
   int d23=index+disp(1,0,0,nx,ny);
   int d24=index+disp(2,0,0,nx,ny);
   int d31=index+disp(-1,0,1,nx,ny);
   int d32=index+disp(0,0,1,nx,ny);
   int d33=index+disp(1,0,1,nx,ny);
   int d34=index+disp(2,0,1,nx,ny);
   int d41=index+disp(-1,0,2,nx,ny);
   int d42=index+disp(0,0,2,nx,ny);
   int d43=index+disp(1,0,2,nx,ny);
   int d44=index+disp(2,0,2,nx,ny);

   if(d11>0 && d44<(nx*ny*nz)){
      f_y_v2n[index] = coef_int[0]  * (coef_der1_stag[0]*f[d11] + coef_der1_stag[1]*f[d12] +
                     coef_der1_stag[2]*f[d13] + coef_der1_stag[3]*f[d14]) + 
                     coef_int[1] * (coef_der1_stag[0]*f[d21] + coef_der1_stag[1]*f[d22] +
                     coef_der1_stag[2]*f[d23] + coef_der1_stag[3]*f[d24])+ 
                     coef_int[2]  * (coef_der1_stag[0]*f[d31] + coef_der1_stag[1]*f[d32] +
                     coef_der1_stag[2]*f[d33] + coef_der1_stag[3]*f[d34]) +
                     coef_int[3]  * (coef_der1_stag[0]*f[d41] + coef_der1_stag[1]*f[d42] +
                     coef_der1_stag[2]*f[d43] + coef_der1_stag[3]*f[d44]);
      f_y_v2n[index] *= fact;
   }

}

__global__ void grady_v2n_fd4(double *f, double *f_y_v2n, int nz, int ny, int nx, double fact){
    grady_v2n_fd4Device(f, f_y_v2n, nz, ny, nx, fact);
}
extern "C" void grady_v2n_fd4_cuda_(double *f, double *f_y_v2n,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   grady_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_y_v2n, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__device__ void gradz_n2v_fd4Device(double *f, double *f_z_n2v, int nz, int ny, int nx, double fact)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-2,0,-2,nx,ny);
   int d12=index+disp(-2,0,-1,nx,ny);
   int d13=index+disp(-2,0,0,nx,ny);
   int d14=index+disp(-2,0,1,nx,ny);
   int d21=index+disp(-1,0,-2,nx,ny);
   int d22=index+disp(-1,0,-1,nx,ny);
   int d23=index+disp(-1,0,0,nx,ny);
   int d24=index+disp(-1,0,1,nx,ny);
   int d31=index+disp(0,0,-2,nx,ny);
   int d32=index+disp(0,0,-1,nx,ny);
   int d33=index+disp(0,0,0,nx,ny);
   int d34=index+disp(0,0,1,nx,ny);
   int d41=index+disp(1,0,-2,nx,ny);
   int d42=index+disp(1,0,-1,nx,ny);
   int d43=index+disp(1,0,0,nx,ny);
   int d44=index+disp(1,0,1,nx,ny);

   if (d11>0 && d44<(nx*ny*nz)){
      f_z_n2v[index] = coef_int[0]  * (coef_der1_stag[0]*f[d11] + coef_der1_stag[1]*f[d12] +
                     coef_der1_stag[2]*f[d13] + coef_der1_stag[3]*f[d14]) + 
                     coef_int[1] * (coef_der1_stag[0]*f[d21] + coef_der1_stag[1]*f[d22] +
                     coef_der1_stag[2]*f[d23] + coef_der1_stag[3]*f[d24])+ 
                     coef_int[2]  * (coef_der1_stag[0]*f[d31] + coef_der1_stag[1]*f[d32] +
                     coef_der1_stag[2]*f[d33] + coef_der1_stag[3]*f[d34]) +
                     coef_int[3] *  (coef_der1_stag[0]*f[d41] + coef_der1_stag[1]*f[d42] +
                     coef_der1_stag[2]*f[d43] + coef_der1_stag[3]*f[d44]);
      f_z_n2v[index] *= fact;
   }

}

__global__ void gradz_n2v_fd4(double *f, double *f_z_n2v, int nz, int ny, int nx, double fact){
    gradz_n2v_fd4Device(f, f_z_n2v, nz, ny, nx, fact);
}
extern "C" void gradz_n2v_fd4_cuda_(double *f, double *f_z_n2v,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradz_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_z_n2v, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__device__ void grady_n2v_fd4Device(double *f, double *f_y_n2v, int nz, int ny, int nx, double fact)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-2,0,-2,nx,ny);
   int d12=index+disp(-1,0,-2,nx,ny);
   int d13=index+disp(0 ,0,-2,nx,ny);
   int d14=index+disp(1 ,0,-2,nx,ny);
   int d21=index+disp(-2,0,-1,nx,ny);
   int d22=index+disp(-1,0,-1,nx,ny);
   int d23=index+disp(0 ,0,-1,nx,ny);
   int d24=index+disp(1 ,0,-1,nx,ny);
   int d31=index+disp(-2,0,0 ,nx,ny);
   int d32=index+disp(-1,0,0 ,nx,ny);
   int d33=index+disp(0 ,0,0 ,nx,ny);
   int d34=index+disp(1 ,0,0 ,nx,ny);
   int d41=index+disp(-2,0,1 ,nx,ny);
   int d42=index+disp(-1,0,1 ,nx,ny);
   int d43=index+disp(0 ,0,1 ,nx,ny);
   int d44=index+disp(1 ,0,1 ,nx,ny);

   if (d11>0 && d44<(nx*ny*nz)){
      f_y_n2v[index] = coef_int[0]  * (coef_der1_stag[0]*f[d11] + coef_der1_stag[1]*f[d12] +
                     coef_der1_stag[2]*f[d13] + coef_der1_stag[3]*f[d14]) + 
                     coef_int[1] * (coef_der1_stag[0]*f[d21] + coef_der1_stag[1]*f[d22] +
                     coef_der1_stag[2]*f[d23] + coef_der1_stag[3]*f[d24])+ 
                     coef_int[2]  * (coef_der1_stag[0]*f[d31] + coef_der1_stag[1]*f[d32] +
                     coef_der1_stag[2]*f[d33] + coef_der1_stag[3]*f[d34]) +
                     coef_int[3] *  (coef_der1_stag[0]*f[d41] + coef_der1_stag[1]*f[d42] +
                     coef_der1_stag[2]*f[d43] + coef_der1_stag[3]*f[d44]);
      f_y_n2v[index] *= fact;
   }

}

__global__ void grady_n2v_fd4(double *f, double *f_y_n2v, int nz, int ny, int nx, double fact){
    grady_n2v_fd4Device(f, f_y_n2v, nz, ny, nx, fact);
}
extern "C" void grady_n2v_fd4_cuda_(double *f, double *f_y_n2v,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   grady_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_y_n2v, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}


__device__ void gradxx_n2n_fd4Device(double *f, double *f_xx, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d1=disp(0,1,0,nx,ny);
   int d2=disp(0,2,0,nx,ny);

   if((index-d2)>=0 && index<(nx*ny*nz-d2)){
      f_xx[index] = coef_der2[0] * f[index-d2] + coef_der2[1]*f[index-d1] +
                    coef_der2[2] * f[index] + coef_der2[3]*f[index+d1] +
                    coef_der2[4] * f[index+d2];
      f_xx[index] *= fact;

   }

}
__global__ void gradxx_n2n_fd4(double *f, double *fy, int nz, int ny, int nx, double fact)
{
   gradxx_n2n_fd4Device(f, fy, nz, ny, nx, fact);
}

extern "C" void gradxx_n2n_fd4_cuda_(double *f, double *f_xx,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradxx_n2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_xx, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__global__ void gradzz_n2n_fd4(double *f, double *f_zz, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d1=disp(0,0,1,nx,ny);
   int d2=disp(0,0,2,nx,ny);

   if((index-d2)>=0 && index<(nx*ny*nz-d2)){
      f_zz[index] = coef_der2[0] * f[index-d2] + coef_der2[1]*f[index-d1] +
                    coef_der2[2] * f[index] + coef_der2[3]*f[index+d1] +
                    coef_der2[4] * f[index+d2];
      f_zz[index] *= fact;

   }

}

extern "C" void gradzz_n2n_fd4_cuda_(double *f, double *f_zz,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradzz_n2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_zz, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__device__ void gradyy_n2n_fd4Device(double *f, double *f_yy, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d1=disp(1,0,0,nx,ny);
   int d2=disp(2,0,0,nx,ny);

   if((index-d2)>=0 && index<(nx*ny*nz-d2)){
      f_yy[index] = coef_der2[0] * f[index-d2] + coef_der2[1]*f[index-d1] +
                    coef_der2[2] * f[index] + coef_der2[3]*f[index+d1] +
                    coef_der2[4] * f[index+d2];
      f_yy[index] *= fact;

   }

}

__global__ void gradyy_n2n_fd4(double *f, double *f_yy, int nz, int ny, int nx, double fact){
    gradyy_n2n_fd4Device(f, f_yy, nz, ny, nx, fact);
}
extern "C" void gradyy_n2n_fd4_cuda_(double *f, double *f_yy,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradyy_n2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_yy, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__global__ void gradyy_v2n_fd4(double *f, double *f_yy_v2n, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-1,0,-1,nx,ny);
   int d12=index+disp(0,0,-1,nx,ny);
   int d13=index+disp(1,0,-1,nx,ny);
   int d14=index+disp(2,0,-1,nx,ny);
   int d21=index+disp(-1,0,0,nx,ny);
   int d22=index+disp(0,0,0,nx,ny);
   int d23=index+disp(1,0,0,nx,ny);
   int d24=index+disp(2,0,0,nx,ny);
   int d31=index+disp(-1,0,1,nx,ny);
   int d32=index+disp(0,0,1,nx,ny);
   int d33=index+disp(1,0,1,nx,ny);
   int d34=index+disp(2,0,1,nx,ny);
   int d41=index+disp(-1,0,2,nx,ny);
   int d42=index+disp(0,0,2,nx,ny);
   int d43=index+disp(1,0,2,nx,ny);
   int d44=index+disp(2,0,2,nx,ny);

   if(d11>0 && d44<(nx*ny*nz)){
      f_yy_v2n[index] = coef_int[0]  * (coef_der2_stag[0]*f[d11] + coef_der2_stag[1]*f[d12] +
                     coef_der2_stag[2]*f[d13] + coef_der2_stag[3]*f[d14]) + 
                     coef_int[1] * (coef_der2_stag[0]*f[d21] + coef_der2_stag[1]*f[d22] +
                     coef_der2_stag[2]*f[d23] + coef_der2_stag[3]*f[d24])+ 
                     coef_int[2]  * (coef_der2_stag[0]*f[d31] + coef_der2_stag[1]*f[d32] +
                     coef_der2_stag[2]*f[d33] + coef_der2_stag[3]*f[d34]) +
                     coef_int[3]  * (coef_der2_stag[0]*f[d41] + coef_der2_stag[1]*f[d42] +
                     coef_der2_stag[2]*f[d43] + coef_der2_stag[3]*f[d44]);
      f_yy_v2n[index] *= fact;
   }

}

extern "C" void gradyy_v2n_fd4_cuda_(double *f, double *f_yy_v2n,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradyy_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_yy_v2n, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__global__ void gradyy_n2v_fd4(double *f, double *f_yy_n2v, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-2,0,-2,nx,ny);
   int d12=index+disp(-1,0,-2,nx,ny);
   int d13=index+disp(0,0,-2,nx,ny);
   int d14=index+disp(1,0,-2,nx,ny);
   int d21=index+disp(-2,0,-1,nx,ny);
   int d22=index+disp(-1,0,-1,nx,ny);
   int d23=index+disp(0,0,-1,nx,ny);
   int d24=index+disp(1,0,-1,nx,ny);
   int d31=index+disp(-2,0,0,nx,ny);
   int d32=index+disp(-1,0,0,nx,ny);
   int d33=index+disp(0,0,0,nx,ny);
   int d34=index+disp(1,0,0,nx,ny);
   int d41=index+disp(-2,0,1,nx,ny);
   int d42=index+disp(-1,0,1,nx,ny);
   int d43=index+disp(0,0,1,nx,ny);
   int d44=index+disp(1,0,1,nx,ny);

   if(d11>0 && d44<(nx*ny*nz)){
      f_yy_n2v[index] = coef_int[0]  * (coef_der2_stag[0]*f[d11] + coef_der2_stag[1]*f[d12] +
                     coef_der2_stag[2]*f[d13] + coef_der2_stag[3]*f[d14]) + 
                     coef_int[1] * (coef_der2_stag[0]*f[d21] + coef_der2_stag[1]*f[d22] +
                     coef_der2_stag[2]*f[d23] + coef_der2_stag[3]*f[d24])+ 
                     coef_int[2]  * (coef_der2_stag[0]*f[d31] + coef_der2_stag[1]*f[d32] +
                     coef_der2_stag[2]*f[d33] + coef_der2_stag[3]*f[d34]) +
                     coef_int[3]  * (coef_der2_stag[0]*f[d41] + coef_der2_stag[1]*f[d42] +
                     coef_der2_stag[2]*f[d43] + coef_der2_stag[3]*f[d44]);
      f_yy_n2v[index] *= fact;
   }

}

extern "C" void gradyy_n2v_fd4_cuda_(double *f, double *f_yy_n2v,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradyy_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_yy_n2v, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__global__ void gradyz_n2v_fd4(double *f, double *f_yz_n2v, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-2,0,-2,nx,ny);
   int d12=index+disp(-2,0,-1,nx,ny);
   int d13=index+disp(-2,0,-0,nx,ny);
   int d14=index+disp(-2,0,1,nx,ny);
   int d21=index+disp(-1,0,-2,nx,ny);
   int d22=index+disp(-1,0,-1,nx,ny);
   int d23=index+disp(-1,0,0,nx,ny);
   int d24=index+disp(-1,0,1,nx,ny);
   int d31=index+disp(0,0,-2,nx,ny);
   int d32=index+disp(0,0,-1,nx,ny);
   int d33=index+disp(0,0,0,nx,ny);
   int d34=index+disp(0,0,1,nx,ny);
   int d41=index+disp(1,0,-2,nx,ny);
   int d42=index+disp(1,0,-1,nx,ny);
   int d43=index+disp(1,0,0,nx,ny);
   int d44=index+disp(1,0,1,nx,ny);

   if(d11>0 && d44<(nx*ny*nz)){
      f_yz_n2v[index] = coef_der1_stag[0]  * (coef_der1_stag[0]*f[d11] + coef_der1_stag[1]*f[d12] +
                     coef_der1_stag[2]*f[d13] + coef_der1_stag[3]*f[d14]) + 
                     coef_der1_stag[1] * (coef_der1_stag[0]*f[d21] + coef_der1_stag[1]*f[d22] +
                     coef_der1_stag[2]*f[d23] + coef_der1_stag[3]*f[d24])+ 
                     coef_der1_stag[2]  * (coef_der1_stag[0]*f[d31] + coef_der1_stag[1]*f[d32] +
                     coef_der1_stag[2]*f[d33] + coef_der1_stag[3]*f[d34]) +
                     coef_der1_stag[3]  * (coef_der1_stag[0]*f[d41] + coef_der1_stag[1]*f[d42] +
                     coef_der1_stag[2]*f[d43] + coef_der1_stag[3]*f[d44]);
      f_yz_n2v[index] *= fact;
   }

}

extern "C" void gradyz_n2v_fd4_cuda_(double *f, double *f_yz_n2v,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradyz_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_yz_n2v, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}

__global__ void gradyz_v2n_fd4(double *f, double *f_yz_v2n, int nz, int ny, int nx, double fact)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-1,0,-1,nx,ny);
   int d12=index+disp(-1,0,0,nx,ny);
   int d13=index+disp(-1,0,1,nx,ny);
   int d14=index+disp(-1,0,2,nx,ny);
   int d21=index+disp(0,0,-1,nx,ny);
   int d22=index+disp(0,0,0,nx,ny);
   int d23=index+disp(0,0,1,nx,ny);
   int d24=index+disp(0,0,2,nx,ny);
   int d31=index+disp(1,0,-1,nx,ny);
   int d32=index+disp(1,0,0,nx,ny);
   int d33=index+disp(1,0,1,nx,ny);
   int d34=index+disp(1,0,2,nx,ny);
   int d41=index+disp(2,0,-1,nx,ny);
   int d42=index+disp(2,0,0,nx,ny);
   int d43=index+disp(2,0,1,nx,ny);
   int d44=index+disp(2,0,2,nx,ny);

   if(d11>0 && d44<(nx*ny*nz)){
      f_yz_v2n[index] = coef_der1_stag[0]  * (coef_der1_stag[0]*f[d11] + coef_der1_stag[1]*f[d12] +
                     coef_der1_stag[2]*f[d13] + coef_der1_stag[3]*f[d14]) + 
                     coef_der1_stag[1] * (coef_der1_stag[0]*f[d21] + coef_der1_stag[1]*f[d22] +
                     coef_der1_stag[2]*f[d23] + coef_der1_stag[3]*f[d24])+ 
                     coef_der1_stag[2]  * (coef_der1_stag[0]*f[d31] + coef_der1_stag[1]*f[d32] +
                     coef_der1_stag[2]*f[d33] + coef_der1_stag[3]*f[d34]) +
                     coef_der1_stag[3]  * (coef_der1_stag[0]*f[d41] + coef_der1_stag[1]*f[d42] +
                     coef_der1_stag[2]*f[d43] + coef_der1_stag[3]*f[d44]);
      f_yz_v2n[index] *= fact;
   }

}

extern "C" void gradyz_v2n_fd4_cuda_(double *f, double *f_yz_v2n,  int *nz, int *ny, int *nx, double *fact){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradyz_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_yz_v2n, *nz, *ny, *nx, *fact);
   cudaDeviceSynchronize();
}
__global__ void scalyy_fd4(double *f, double *diff, double *f_yy, int nz, int ny, int nx, double fact, double *grady_f, double *grady_diff, double fact1)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-2,0,0,nx,ny);
   int d12=index+disp(-1,0,0,nx,ny);
   int d13=index+disp(0,0,0,nx,ny);
   int d14=index+disp(1,0,0,nx,ny);
   int d15=index+disp(2,0,0,nx,ny);
 
   gradyDevice(f, grady_f, nz, ny, nx, fact1);   
   gradyDevice(diff, grady_diff, nz, ny, nx, fact1);   
   __syncthreads();
  
   if(d11>=0 && d15<(nx*ny*nz)){
      f_yy[index] = diff[index] *(coef_der2[0]*f[d11] +
                       coef_der2[1]*f[d12] + coef_der2[2]*f[d13] +
                       coef_der2[3]*f[d14] + coef_der2[4]*f[d15] );
      f_yy[index] *= fact;
      f_yy[index] += grady_diff[index]*grady_f[index];
   }
}

extern "C" void scalyy_fd4_cuda_(double *f, double *diff, double *f_yy, int *nz, int *ny, int *nx, double *fact, double *grady_f, double *grady_diff, double *fact1){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   scalyy_fd4<<<numBlocks, threadsPerBlock>>>( f,  diff, f_yy, *nz, *ny, *nx,  *fact, grady_f, grady_diff,*fact1);
   cudaDeviceSynchronize();
}

__global__ void scalxx_fd4(double *f, double *diff, double *f_xx, int nz, int ny, int nx, double fact, double *gradx_f, double *gradx_diff, double fact1)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(0,-2,0,nx,ny);
   int d12=index+disp(0,-1,0,nx,ny);
   int d13=index+disp(0,0,0,nx,ny);
   int d14=index+disp(0,1,0,nx,ny);
   int d15=index+disp(0,2,0,nx,ny);
 
   gradx_n2n_fd4Device(f, gradx_f, nz, ny, nx, fact1);   
   gradx_n2n_fd4Device(diff, gradx_diff, nz, ny, nx, fact1);   
   __syncthreads();
  
   if(d11>=0 && d15<(nx*ny*nz)){
      f_xx[index] = diff[index] *(coef_der2[0]*f[d11] +
                       coef_der2[1]*f[d12] + coef_der2[2]*f[d13] +
                       coef_der2[3]*f[d14] + coef_der2[4]*f[d15] );
      f_xx[index] *= fact;
      f_xx[index] += gradx_diff[index]*gradx_f[index];
   }
}

extern "C" void scalxx_fd4_cuda_(double *f, double *diff, double *f_xx, int *nz, int *ny, int *nx, double *fact, double *gradx_f, double *gradx_diff, double *fact1){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   scalxx_fd4<<<numBlocks, threadsPerBlock>>>( f,  diff, f_xx, *nz, *ny, *nx,  *fact, gradx_f, gradx_diff,*fact1);
   cudaDeviceSynchronize();
}

__global__ void lapl_fd4(double *f,  double *f_lapl, int nz, int ny, int nx,  double *f_xx, double *f_yy, double fact, double fact1)
{
   int index = blockDim.x*blockIdx.x+threadIdx.x;

 
   gradxx_n2n_fd4Device(f, f_xx, nz, ny, nx, fact);   
   gradyy_n2n_fd4Device(f, f_yy, nz, ny, nx, fact1);   
   __syncthreads();
  
   if(index<(nx*ny*nz)){
      f_lapl[index] = f_xx[index]+f_yy[index];
   }
}

extern "C" void lapl_fd4_cuda_(double *f, double *f_lapl, int *nz, int *ny, int *nx, double *f_xx, double *f_yy, double *fact, double *fact1){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   lapl_fd4<<<numBlocks, threadsPerBlock>>>( f, f_lapl, *nz,  *ny,  *nx, f_xx, f_yy, *fact, *fact1);
   cudaDeviceSynchronize();
}

__device__ void interp_v2n_fd4Device(double *f, double *f_int_v2n, int nz, int ny, int nx)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-1,0,-1,nx,ny);
   int d12=index+disp(-1,0,0,nx,ny);
   int d13=index+disp(-1,0,1,nx,ny);
   int d14=index+disp(-1,0,2,nx,ny);
   int d21=index+disp(0,0,-1,nx,ny);
   int d22=index+disp(0,0,0,nx,ny);
   int d23=index+disp(0,0,1,nx,ny);
   int d24=index+disp(0,0,2,nx,ny);
   int d31=index+disp(1,0,-1,nx,ny);
   int d32=index+disp(1,0,0,nx,ny);
   int d33=index+disp(1,0,1,nx,ny);
   int d34=index+disp(1,0,2,nx,ny);
   int d41=index+disp(2,0,-1,nx,ny);
   int d42=index+disp(2,0,0,nx,ny);
   int d43=index+disp(2,0,1,nx,ny);
   int d44=index+disp(2,0,2,nx,ny);

   if (d11>0 && d44<(nx*ny*nz)){
      f_int_v2n[index] = coef_int[0]  * (coef_int[0]*f[d11] + coef_int[1]*f[d12] +
                     coef_int[2]*f[d13] + coef_int[3]*f[d14]) + 
                     coef_int[1] * (coef_int[0]*f[d21] + coef_int[1]*f[d22] +
                     coef_int[2]*f[d23] + coef_int[3]*f[d24])+ 
                     coef_int[2]  * (coef_int[0]*f[d31] + coef_int[1]*f[d32] +
                     coef_int[2]*f[d33] + coef_int[3]*f[d34]) +
                     coef_int[3] *  (coef_int[0]*f[d41] + coef_int[1]*f[d42] +
                     coef_int[2]*f[d43] + coef_int[3]*f[d44]);
   }

}

__global__ void interp_v2n_fd4(double *f, double *f_int_v2n, int nz, int ny, int nx){
    interp_v2n_fd4Device(f, f_int_v2n, nz, ny, nx);
}

extern "C" void interp_v2n_fd4_cuda_(double *f, double *f_int_v2n,  int *nz, int *ny, int *nx){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   interp_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_int_v2n, *nz, *ny, *nx);
   cudaDeviceSynchronize();
}

__device__ void interp_n2v_fd4Device(double *f, double *f_int_n2v, int nz, int ny, int nx)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int d11=index+disp(-2,0,-2,nx,ny);
   int d12=index+disp(-2,0,-1,nx,ny);
   int d13=index+disp(-2,0,0,nx,ny);
   int d14=index+disp(-2,0,1,nx,ny);
   int d21=index+disp(-1,0,-2,nx,ny);
   int d22=index+disp(-1,0,-1,nx,ny);
   int d23=index+disp(-1,0,0,nx,ny);
   int d24=index+disp(-1,0,1,nx,ny);
   int d31=index+disp(0,0,-2,nx,ny);
   int d32=index+disp(0,0,-1,nx,ny);
   int d33=index+disp(0,0,0,nx,ny);
   int d34=index+disp(0,0,1,nx,ny);
   int d41=index+disp(1,0,-2,nx,ny);
   int d42=index+disp(1,0,-1,nx,ny);
   int d43=index+disp(1,0,0,nx,ny);
   int d44=index+disp(1,0,1,nx,ny);

   if (d11>0 && d44<(nx*ny*nz)){
      f_int_n2v[index] = coef_int[0]  * (coef_int[0]*f[d11] + coef_int[1]*f[d12] +
                     coef_int[2]*f[d13] + coef_int[3]*f[d14]) + 
                     coef_int[1] * (coef_int[0]*f[d21] + coef_int[1]*f[d22] +
                     coef_int[2]*f[d23] + coef_int[3]*f[d24])+ 
                     coef_int[2]  * (coef_int[0]*f[d31] + coef_int[1]*f[d32] +
                     coef_int[2]*f[d33] + coef_int[3]*f[d34]) +
                     coef_int[3] *  (coef_int[0]*f[d41] + coef_int[1]*f[d42] +
                     coef_int[2]*f[d43] + coef_int[3]*f[d44]);
   }

}

__global__ void interp_n2v_fd4(double *f, double *f_int_n2v, int nz, int ny, int nx){
    interp_n2v_fd4Device(f, f_int_n2v, nz, ny, nx);
}

extern "C" void interp_n2v_fd4_cuda_(double *f, double *f_int_n2v,  int *nz, int *ny, int *nx){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   interp_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_int_n2v, *nz, *ny, *nx);
   cudaDeviceSynchronize();
}

__global__ void gradpar_v2n_fd4(double *f, double *f_grad, int nz, int ny, int nx,  double *f_z, double *f_y, double *f_n, double *f_x_n, double deltazi, double deltayi, double deltaxi, double gradpar_z, double *gradpar_x_n, double *gradpar_y_n)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int ind=xyindex(index, nx*ny);

   gradz_v2n_fd4Device(f, f_z, nz, ny, nx, deltazi); 
   grady_v2n_fd4Device(f, f_y, nz, ny, nx, deltayi); 
   interp_v2n_fd4Device(f, f_n, nz, ny, nx); 
   __syncthreads();
   if (index<(nx)*(ny)*(nz)){
      f_grad[index] = gradpar_z*f_z[index] + gradpar_y_n[ind]*f_y[index] ;
   }

   /*gradx_n2n_fd4Device(f_n, f_x_n, nz, ny, nx, deltaxi); 
   __syncthreads();
   
   if (index<(nx)*(ny)*(nz)){
      f_grad[index] +=    gradpar_x_n[ind]*f_x_n[index];
   }*/

}

__global__ void gradpar1_v2n_fd4(double *f, double *f_grad, int nz, int ny, int nx,  double *f_z, double *f_y, double *f_n, double *f_x_n, double deltazi, double deltayi, double deltaxi, double gradpar_z, double *gradpar_x_n, double *gradpar_y_n)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int ind=xyindex(index, nx*ny);
   gradx_n2n_fd4Device(f_n, f_x_n, nz, ny, nx, deltaxi); 
   __syncthreads();
   
   if (index<(nx)*(ny)*(nz)){
      f_grad[index] += gradpar_x_n[ind]*f_x_n[index];
   }



}


extern "C" void gradpar_v2n_fd4_cuda_(double *f, double *f_grad, int *nz, int *ny, int *nx,  double *f_z, double *f_y, double *f_n, double *f_x_n, double *deltazi, double *deltayi, double *deltaxi, double *gradpar_z, double *gradpar_x_n, double *gradpar_y_n){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 1024;
   dim3 numBlocks = ((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradpar_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_grad, *nz, *ny, *nx, f_z, f_y, f_n, f_x_n, *deltazi, *deltayi, *deltaxi, *gradpar_z, gradpar_x_n, gradpar_y_n);
   cudaDeviceSynchronize();
   gradpar1_v2n_fd4<<<numBlocks, threadsPerBlock>>>(f, f_grad, *nz, *ny, *nx, f_z, f_y, f_n, f_x_n, *deltazi, *deltayi, *deltaxi, *gradpar_z, gradpar_x_n, gradpar_y_n);
   cudaDeviceSynchronize();
}

__global__ void gradpar_n2v_fd4(double *f, double *f_grad, int nz, int ny, int nx,  double *f_z, double *f_y, double *f_v, double *f_x_v, double deltazi, double deltayi, double deltaxi, double gradpar_z, double *gradpar_x_v, double *gradpar_y_v)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int ind=xyindex(index, nx*ny);

   gradz_n2v_fd4Device(f, f_z, nz, ny, nx, deltazi); 
   grady_n2v_fd4Device(f, f_y, nz, ny, nx, deltayi); 
   interp_n2v_fd4Device(f, f_v, nz, ny, nx); 
   __syncthreads();
   //gradx_n2n_fd4Device(f_v, f_x_v, nz, ny, nx, deltaxi); 
   //__syncthreads();
   
   if (index<(nx)*(ny)*(nz)){
      f_grad[index] = gradpar_z*f_z[index] + gradpar_y_v[ind]*f_y[index];// +  gradpar_x_v[ind]*f_x_n[index];
   }

}

__global__ void gradpar1_n2v_fd4(double *f, double *f_grad, int nz, int ny, int nx,  double *f_z, double *f_y, double *f_v, double *f_x_v, double deltazi, double deltayi, double deltaxi, double gradpar_z, double *gradpar_x_v, double *gradpar_y_v)
{
   int index=blockDim.x*blockIdx.x+threadIdx.x;
   int ind=xyindex(index, nx*ny);

   gradx_n2n_fd4Device(f_v, f_x_v, nz, ny, nx, deltaxi); 
   __syncthreads();
   
   if (index<(nx)*(ny)*(nz)){
      //f_grad[index] +=   gradpar_x_v[ind]*f_x_v[index];
      f_grad[index] =   f_v[index];
   }

}


extern "C" void gradpar_n2v_fd4_cuda_(double *f, double *f_grad, int *nz, int *ny, int *nx,  double *f_z, double *f_y, double *f_v, double *f_x_v, double *deltazi, double *deltayi, double *deltaxi, double *gradpar_z, double *gradpar_x_v, double *gradpar_y_v){
   int n = (*nx)*(*ny)*(*nz);
   dim3 threadsPerBlock = 256;
   dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x);
   gradpar_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_grad, *nz, *ny, *nx, f_z, f_y, f_v, f_x_v, *deltazi, *deltayi, *deltaxi, *gradpar_z, gradpar_x_v, gradpar_y_v);
   cudaDeviceSynchronize();
   gradpar1_n2v_fd4<<<numBlocks, threadsPerBlock>>>(f, f_grad, *nz, *ny, *nx, f_z, f_y, f_v, f_x_v, *deltazi, *deltayi, *deltaxi, *gradpar_z, gradpar_x_v, gradpar_y_v);
   cudaDeviceSynchronize();
}
