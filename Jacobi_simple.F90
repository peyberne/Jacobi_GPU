program jacobi
#ifdef CUDA
  use cuda_interface
  use iso_fortran_env, only: di=>int64
#else
  use iso_fortran_env, only: di=>int64, dp=>real64
#endif
#ifdef CUDAFOR
use cudafor
use jacobi_cufor
#endif
#ifdef BW
use bw_cufor
#endif
  implicit none

  integer,  parameter :: iter_max = 10
  integer,  parameter :: N = 8*1024
  integer             ::  i,j,iterk,istat
  real(dp), parameter :: tol = 1.0e-2_dp
  
  real(dp), allocatable,target  :: A(:, :), Anew(:, :)
  real(dp), allocatable,target  :: A_omp(:, :), Anew_omp(:, :)
  real(dp), pointer, save       :: A_cuda(:,:), Anew_cuda(:,:)
  real(dp), pointer             :: A_pointer1(:,:)
  real(dp), pointer             :: A_pointer2(:,:)
#ifdef CUDAFOR
  real(dp), allocatable,managed :: A_cufor(:, :), Anew_cufor(:, :)
  integer, parameter :: threadsPerBlock=256
  integer :: numBlocks
#endif
  real(dp) :: rate
  integer(di)  :: startc, endc
  logical :: isDevice
#ifdef BW
  real(dp), managed :: AA(N*N)
  real(dp), managed :: BB(N*N)
  real(dp), managed :: CC(N*N)
  real(dp) :: rnd
#endif


  call system_clock(count_rate=rate)

#ifdef BW
  rnd=3.42
  AA(1)=0.12
  BB(1)=0.23
  do i=2,N*N
     AA(i)= AA(i-1)+0.000342
     BB(i)= 2.1*AA(i)/1.3
  end do
  call system_clock(startc)
  do iterk=1,iter_max
     call BWdevice<<<(N*N+256-1)/256,256>>>(AA, BB, CC, N*N)
  enddo
  istat = cudaDeviceSynchronize()
  call system_clock(endc)
  write(*,*) "Bandwidth in GBytes/s: ", iter_max*N*N*24.*real(endc-startc, dp)/rate/1000000000.
#endif



  allocate(A(N, N))
  allocate(Anew(N, N))

! Jacobi on CPU
  call init_array(A, N, 7.0_dp)
  call init_array(Anew, N, 7.0_dp)
  A_pointer1 => A
  A_pointer2 => Anew
  call system_clock(startc)
  do iterk=1,iter_max
     call jacobi_CPU(A_pointer1, A_pointer2, N)
     call swap(A_pointer1,A_pointer2)
   enddo
  call system_clock(endc)
  write(*,*) "timing for Jacobi CPU: ", real(endc-startc, dp)/rate*1000, "ms"
! Jacobi in cuda
#ifdef CUDA
  call gbs_allocate_cuda(A_cuda,1,N,1,N)  
  call gbs_allocate_cuda(Anew_cuda,1,N,1,N)    
  call init_array_cuda(A_cuda, N, 7.0_dp)
  call init_array_cuda(Anew_cuda, N, 7.0_dp)
  call system_clock(startc)
  do iterk=1,iter_max
     if (mod(iterk,2).eq.1) then
        CALL jacobi_cuda_opt(A_cuda,Anew_cuda,N,N)
     else
        CALL jacobi_cuda_opt(Anew_cuda,A_cuda,N,N)
     endif
  enddo
  CALL synchronize_cuda_device()
  call system_clock(endc)
  write(*,*) "timing for Jacobi GPU cuda: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array_cuda(A_cuda,A,N)
#endif
!
#ifndef CUDA
  allocate(A_omp(N, N))
  allocate(Anew_omp(N, N))
! Jacobi openmp 
  call init_array(A_omp, N, 7.0_dp)
  call init_array(Anew_omp, N, 7.0_dp)
  call system_clock(startc)
  call jacobi_openmp(A_omp, Anew_omp, N, tol, iter_max)
  call system_clock(endc)
  write(*,*) "timing for Jacobi CPU openmp: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array(A_omp,A,N)

! Jacobi openmp offload
  call init_array(A_omp, N, 7.0_dp)
  call init_array(Anew_omp, N, 7.0_dp)
  A_pointer1 => A_omp
  A_pointer2 => Anew_omp
  !$omp target enter data map(to: A_omp, Anew_omp)
  call system_clock(startc)
  do iterk=1,iter_max
     call jacobi_offload(A_pointer1, A_pointer2, N)
     call swap(A_pointer1,A_pointer2)
  enddo
  call system_clock(endc)
  write(*,*) "timing for Jacobi GPU openmp: ", real(endc-startc, dp)/rate*1000, "ms"
  !$omp target update from(A_omp)  
  call test_array(A_omp,A,N)
#endif
!
! Jacobi cudafor 
#ifdef CUDAFOR
  allocate(A_cufor(N,N))
  allocate(Anew_cufor(N,N))
  call init_array(A_cufor, N, 7.0_dp)
  call init_array(Anew_cufor, N, 7.0_dp)
  numBlocks=((N*N+threadsPerBlock-1)/threadsPerBlock)
  call system_clock(startc)
  do iterk=1,iter_max
     if (mod(iterk,2).eq.1) then
        call jacobiDevice<<<numBlocks,threadsPerBlock>>>(A_cufor, Anew_cufor, N)
     else
        call jacobiDevice<<<numBlocks,threadsPerBlock>>>(Anew_cufor, A_cufor, N)
     endif
  enddo
  istat = cudaDeviceSynchronize()
  call system_clock(endc)
  write(*,*) "timing for Jacobi GPU CUDAFOR: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array(A_cufor,A,N)
#endif
!
contains


  subroutine init_array(Tab, N, val)
    real(dp), intent(inout), allocatable :: Tab(:,:)
    integer,  intent(in)                 :: N
    real(dp), intent(in)                 :: val

     integer i, j

    Tab = 0.0_dp
    
    do i = 1, N
       Tab(1, i) = val
       Tab(N, i) = val/2.0_dp
       Tab(i, 1) = val/3.0_dp
       Tab(i, N) = val/5.0_dp
    end do
  end subroutine init_array

  subroutine init_array_cuda(A_cuda, N, val)
    real(dp), intent(inout), pointer     :: A_cuda(:,:)
    integer,  intent(in)                 :: N
    real(dp), intent(in)                 :: val

    integer i, j

    A_cuda = 0.0_dp
    
    do i = 1, N
       A_cuda(1, i) = val
       A_cuda(N, i) = val/2.0_dp
       A_cuda(i, 1) = val/3.0_dp
       A_cuda(i, N) = val/5.0_dp
    end do
  end subroutine init_array_cuda

  subroutine test_array(Tab, Aref, N)
    real(dp), intent(in), allocatable :: Tab(:,:)
    real(dp), intent(in), allocatable :: Aref(:,:)
    integer,  intent(in)                 :: N

    integer i, j

    do j = 2, N-1
       do i = 2, N-1
          if (abs((Tab(i, j) - Aref(i, j))/Aref(i, j)) > 100.0 * epsilon(1.0_dp)) then
             write(*,*) "Error: ", abs((Tab(i, j) - Aref(i, j))/Aref(i, j))
             return
          end if
       end do
    end do
  end subroutine test_array

  subroutine test_array_cuda(A_cuda, Aref, N)
    real(dp), intent(in), pointer :: A_cuda(:,:)
    real(dp), intent(in), allocatable :: Aref(:,:)
    integer,  intent(in)                 :: N

    integer i, j

    do j = 2, N-1
       do i = 2, N-1
          if (abs((A_cuda(i, j) - Aref(i, j))/Aref(i, j)) > 100. * epsilon(1.0_dp)) then
             write(*,*) "Error: ", abs((A_cuda(i, j) - Aref(i, j))/Aref(i, j))
             return
          end if
       end do
    end do
  end subroutine test_array_cuda

  subroutine jacobi_CPU(Tab, TabNew, N)
    real(dp), intent(in)       :: Tab(N,N)
    real(dp), intent(inout)    :: TabNew(N,N)
    integer,  intent(in)       :: N
    integer i, j

    do j = 2, N-1
       do i = 2, N-1
          TabNew(i, j) = 0.25 * (Tab(i, j+1) + Tab(i, j-1) + Tab(i+1, j) + Tab(i-1, j))
       end do
    end do

  end subroutine jacobi_CPU
  
  subroutine jacobi_offload(Tab, TabNew, N)
    real(dp), intent(in)       :: Tab(N,N)
    real(dp), intent(inout)    :: TabNew(N,N)
    integer,  intent(in)       :: N
    integer                    :: i, j

!$omp target teams distribute parallel do simd collapse(2) 
     do j = 2, N-1
        do i = 2, N-1
          TabNew(i, j) = 0.25 * (Tab(i, j+1) + Tab(i, j-1) + Tab(i+1, j) + Tab(i-1, j))
       end do
    end do
!$omp end target teams distribute parallel do simd

  end subroutine jacobi_offload

  subroutine jacobi_openmp(Tab, Tabnew, N, tol , iter_max)
    real(dp), intent(inout),target       :: Tab(N,N)
    real(dp), intent(inout), target      :: TabNew(N,N)
    real(dp), pointer                    :: Tab_pointer1(:,:)
    real(dp), pointer                    :: Tab_pointer2(:,:)
    integer,  intent(in)                 :: N
    real(dp), intent(in)                 :: tol
    integer,  intent(in)                 :: iter_max

    integer i, j, iter,k
    real(dp) :: err
    Tab_pointer1 => Tab
    Tab_pointer2 => TabNew
    do iter=1,iter_max
    !$omp parallel
       !$omp do 
       do j = 2, N-1
          !$omp simd
          do i = 2, N-1
             Tab_pointer2(i, j) = 0.25 * (Tab_pointer1(i, j+1) + Tab_pointer1(i, j-1) + Tab_pointer1(i+1, j) + Tab_pointer1(i-1, j))
          end do
          !$omp end simd
       end do
       !$omp end do
    !$omp end parallel
       call swap(Tab_pointer1,Tab_pointer2)
    enddo
    
  end subroutine jacobi_openmp
  subroutine swap(Tab_pointer1,Tab_pointer2)
    real(dp), pointer, intent(inout)  :: Tab_pointer1(:,:)
    real(dp), pointer, intent(inout)  :: Tab_pointer2(:,:)
    real(dp), pointer                 :: tmp_pointer(:,:)
!!$omp declare target
    tmp_pointer  => Tab_pointer1
    Tab_pointer1 => Tab_pointer2
    Tab_pointer2 => tmp_pointer
  end subroutine swap
end program jacobi
