module jacobi_cufor
  use cudafor
  use iso_fortran_env, only: di=>int64, dp=>real64
  contains

  attributes(global) subroutine jacobiDevice(A_cufor,Anew_cufor,N)
      integer, value     :: N
      real(dp), managed  :: A_cufor(N,N)
      real(dp), managed  :: Anew_cufor(N,N)
      integer            :: index,ind1,ind2
      
!EX: write the kernel for 1D blocks using the xindex and yindex functions
!     to map 1d index to 2D indexes (ind1,ind2)   

  end subroutine

  attributes(global) subroutine jacobiDevice2d(A_cufor,Anew_cufor,N)
      integer, value     :: N
      real(dp), managed  :: A_cufor(N,N)
      real(dp), managed  :: Anew_cufor(N,N)
      integer            :: ind1,ind2
       
!EX: write the kernel using the (ind1,ind2) indexes of the 2D block

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


program jacobi
  use iso_fortran_env, only: di=>int64, dp=>real64
use cudafor
use jacobi_cufor
  implicit none

  integer,  parameter :: iter_max = 10 ! number of Jacobi iterations 
  integer,  parameter :: N = 8*1024 ! dimension of x and y arrays
  integer             ::  i,j,iterk,istat
  
  real(dp), allocatable,target  :: A(:, :), Anew(:, :)
  real(dp), pointer             :: A_pointer1(:,:)
  real(dp), pointer             :: A_pointer2(:,:)
!
!EX : Modify allocatable arrays for cuda fortran (A_cufor, Anew_cufor) and 
!      theirs pointers Acufor_pointer1, Acufor_pointer2 using managed attributes
!
  real(dp), allocatable,target :: A_cufor(:, :), Anew_cufor(:, :)
  real(dp), pointer    :: Acufor_pointer1(:,:)
  real(dp), pointer    :: Acufor_pointer2(:,:)
  integer, parameter :: threadsPerBlock=256 ! number of threads per block
!
  integer :: numBlocks 
  type(dim3) :: dimGrid, dimBlock
  real(dp) :: rate
  integer(di)  :: startc, endc
  logical :: isDevice

  call system_clock(count_rate=rate)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi on CPU
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  allocate(A(N, N))
  allocate(Anew(N, N))
  call init_array(A, N, 7.0_dp) !array initialization with boundary conditions
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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi cudafortran with 1d blocks
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  allocate(A_cufor(N,N))
  allocate(Anew_cufor(N,N))
!
  call init_array(A_cufor, N, 7.0_dp)!array initialization with boundary conditions
  call init_array(Anew_cufor, N, 7.0_dp)
!
!EX : compute the number of 1D blocks according to the number of threads per block (threadsPerBlock)
! 
!  numBlocks=...
!
  Acufor_pointer1 => A_cufor
  Acufor_pointer2 => Anew_cufor
  call system_clock(startc)
  do iterk=1,iter_max
!
!EX : call the kernel
!     call jacobiDevice......
!
     call swap(Acufor_pointer1,Acufor_pointer2)!swap the pointer to avoid copy back
  enddo
!
!EX : write synchronization host-device before using A_cufor array
!
  call system_clock(endc)
  write(*,*) "timing for Jacobi GPU CUDAFOR: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array(A_cufor,A,N)!test results
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Jacobi cudafortran with 2d blocks
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  call init_array(A_cufor, N, 7.0_dp)
  call init_array(Anew_cufor, N, 7.0_dp)
!
!EX : compute the dimension of the block (dimBlock) and the dimension of the grid (dimGrid)
!  dimBlock = ...
!  dimGrid = ...
!
  Acufor_pointer1 => A_cufor
  Acufor_pointer2 => Anew_cufor
  call system_clock(startc)
  do iterk=1,iter_max
!
!EX : call the kernel with 2d blocks
!     call jacobiDevice2d......
!
     call swap(Acufor_pointer1,Acufor_pointer2)
  enddo
!
!EX : write synchronization host-device before using A_cufor array
!
  call system_clock(endc)
  write(*,*) "timing for Jacobi GPU CUDAFOR 2D blocks: ", real(endc-startc, dp)/rate*1000, "ms"
  call test_array(A_cufor,A,N)
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

