program jacobi
  use iso_fortran_env, only: di=>int64, dp=>real64
#ifdef CUDAFOR
use cudafor
use jacobi_cufor
#endif
use omp_lib
  implicit none

  integer,  parameter :: iter_max = 10
  integer,  parameter :: N = 4*1024
  integer             ::  i,j,iterk,istat
  
  real(dp), allocatable,target  :: A(:, :), Anew(:, :)
  real(dp), pointer             :: A_pointer1(:,:)
  real(dp), pointer             :: A_pointer2(:,:)
  real(dp), managed, pointer    :: Acufor_pointer1(:,:)
  real(dp), managed, pointer    :: Acufor_pointer2(:,:)
#ifdef CUDAFOR
  real(dp), allocatable,managed,target :: A_cufor(:, :), Anew_cufor(:, :)
  integer, parameter :: threadsPerBlock=256
  integer :: numBlocks
  type(dim3) :: dimGrid, dimBlock
#endif
  real(dp) :: rate
  integer(di)  :: startc, endc
  logical :: isDevice

  call system_clock(count_rate=rate)

  allocate(A(N, N))
  allocate(Anew(N, N))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi on CPU with pointer
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  write(*,*) "timing for Jacobi with pointer CPU: ", real(endc-startc, dp)/rate*1000, "ms"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi on CPU without pointer
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  call init_array(A, N, 7.0_dp)
  call init_array(Anew, N, 7.0_dp)
  call system_clock(startc)
  do iterk=1,iter_max
     if (mod(iterk,2).eq.1) then
        call jacobi_CPU(A, Anew, N)
     else
        call jacobi_CPU(Anew, A, N)
     endif
   enddo
  call system_clock(endc)
  write(*,*) "timing for Jacobi without pointer CPU: ", real(endc-startc, dp)/rate*1000, "ms"
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi cudafortran without pointer 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  write(*,*) "timing for Jacobi without pointer GPU CUDAFOR: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array(A_cufor,A,N)
#endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi cudafortran with pointer 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef CUDAFOR
  call init_array(A_cufor, N, 7.0_dp)
  call init_array(Anew_cufor, N, 7.0_dp)
  numBlocks=((N*N+threadsPerBlock-1)/threadsPerBlock)
  Acufor_pointer1 => A_cufor
  Acufor_pointer2 => Anew_cufor
  call system_clock(startc)
  do iterk=1,iter_max
     call jacobiDevice<<<numBlocks,threadsPerBlock>>>(Acufor_pointer1, Acufor_pointer2, N)
     call swap(Acufor_pointer1,Acufor_pointer2)
  enddo
  istat = cudaDeviceSynchronize()
  call system_clock(endc)
  write(*,*) "timing for Jacobi with pointer GPU CUDAFOR: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array(A_cufor,A,N)
#endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi cudafortran 2D blocks with pointer 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef CUDAFOR
  call init_array(A_cufor, N, 7.0_dp)
  call init_array(Anew_cufor, N, 7.0_dp)
  dimBlock = dim3(16,16,1)
  dimGrid = dim3(ceiling(real(N)/dimBlock%x), ceiling(real(N)/dimBlock%y), 1)
  Acufor_pointer1 => A_cufor
  Acufor_pointer2 => Anew_cufor
  call system_clock(startc)
  do iterk=1,iter_max
     call jacobiDevice2d<<<dimGrid,dimBlock>>>(Acufor_pointer1, Acufor_pointer2, N)
     call swap(Acufor_pointer1,Acufor_pointer2)
  enddo
  istat = cudaDeviceSynchronize()
  call system_clock(endc)
  write(*,*) "timing for Jacobi with pointer GPU CUDAFOR 2D blocks: ", real(endc-startc, dp)/rate*1000, "ms"
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
  
  subroutine swap(Tab_pointer1,Tab_pointer2)
    real(dp), pointer, intent(inout)  :: Tab_pointer1(:,:)
    real(dp), pointer, intent(inout)  :: Tab_pointer2(:,:)
    real(dp), pointer                 :: tmp_pointer(:,:)
    tmp_pointer  => Tab_pointer1
    Tab_pointer1 => Tab_pointer2
    Tab_pointer2 => tmp_pointer
  end subroutine swap

end program jacobi

module jacobi_cufor
  use cudafor
  use iso_fortran_env, only: di=>int64, dp=>real64
  contains

  attributes(global) subroutine jacobiDevice(A_cufor,Anew_cufor,N)
      integer, value     :: N
      real(dp), managed  :: A_cufor(N,N)
      real(dp), managed  :: Anew_cufor(N,N)
      integer            :: index,ind1,ind2
       
      index = blockDim%x*(blockIdx%x-1)+threadIdx%x
      ind1=xindex(index,N)
      ind2=yindex(index,N)
      if ((ind2.NE.1).AND.(ind2.NE.N).AND.(ind1.NE.1).AND.(ind1.NE.N).AND.(index.LE.(N*N))) then
         Anew_cufor(ind1,ind2) = A_cufor(ind1,ind2-1) + A_cufor(ind1,ind2+1) + A_cufor(ind1-1,ind2) + A_cufor(ind1+1,ind2)
         Anew_cufor(ind1,ind2) = Anew_cufor(ind1,ind2) * 0.25
      endif

  end subroutine

  attributes(global) subroutine jacobiDevice2d(A_cufor,Anew_cufor,N)
      integer, value     :: N
      real(dp), managed  :: A_cufor(N,N)
      real(dp), managed  :: Anew_cufor(N,N)
      integer            :: ind1,ind2
       
      ind1 = blockDim%x*(blockIdx%x-1)+threadIdx%x
      ind2 = blockDim%y*(blockIdx%y-1)+threadIdx%y
      if ((ind2.NE.1).AND.(ind2.NE.N).AND.(ind1.NE.1).AND.(ind1.NE.N).AND.(index.LE.(N*N))) then
         Anew_cufor(ind1,ind2) = A_cufor(ind1,ind2-1) + A_cufor(ind1,ind2+1) + A_cufor(ind1-1,ind2) + A_cufor(ind1+1,ind2)
         Anew_cufor(ind1,ind2) = Anew_cufor(ind1,ind2) * 0.25
      endif

  end subroutine

  attributes(device) integer function xindex(index,nx)
  integer :: index,nx
  xindex=mod(index-1,nx)+1
  end function

  attributes(device) integer function yindex(index,nx)
  integer :: index,nx
  yindex=(index-1)/nx+1
  end function

end module 
