FC=gfortran
#FC=nvfortran
NVFC=nvfortran
CU=nvcc
EXEC=Jacobi_cuda
FLAGS = -O3 
OMPFLAGS= -O3 -fopenmp -foffload=-lgfortran -lm -foffload="-lm"
CUDAFORFLAGS=-Mcuda -O3 
CUDA=CUDA
CUDAFOR=CUDAFOR
all: $(EXEC)

Jacobi_cuda: fortfiles.o cudatest.o fortest.o 
	$(CU) -o $(EXEC) $(FLAGS) -lgfortran -lcuda Jacobi_simple.o prec_const_mod.o jacobi.o cuda_interface_mod.o cudalloc.o
Jacobi_offload: Jacobi_simple.F90
	$(FC) $(OMPFLAGS) Jacobi_simple.F90 -o Jacobi_offload
Jacobi_cudafor: Jacobi_simple.F90 jacobi_cufor.cuf
	$(NVFC) $(CUDAFORFLAGS) -D$(CUDAFOR) jacobi_cufor.cuf Jacobi_simple.F90 -o Jacobi_cudafor
fortest.o: Jacobi_simple.F90
	$(FC) -c $(FLAGS) -D$(CUDA) Jacobi_simple.F90
cudatest.o: cudalloc.cu cudalloc.h jacobi.cu
	$(CU) $(FLAGS) -c cudalloc.cu jacobi.cu
fortfiles.o: cuda_interface_mod.F90 prec_const_mod.F90
	$(FC) $(FLAGS) -c prec_const_mod.F90 cuda_interface_mod.F90
clean:
	rm -f $(EXEC) Jacobi_simple.o prec_const_mod.o jacobi.o cuda_interface_mod.o cudalloc.o Jacobi_offload Jacobi_cudafor  *.mod *~
