# Jacobi_cuda
implementation of a Jacobi method for GPUs

# to compile and run CUDA implementation:
make
./Jacobi_cuda

# to compile run OpenMP with or without Offload implementation:
make Jacobi_offload
./Jacobi_offload

# to compile run CUDA FORTRAN with or without Offload implementation:
make Jacobi_cudafor
./Jacobi_cudafor

