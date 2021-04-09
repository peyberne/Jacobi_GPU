program jacobi
  use iso_fortran_env, only: di=>int64, dp=>real64
  implicit none

  integer,  parameter :: iter_max = 10 ! number of Jacobi iterations 
  integer,  parameter :: N = 4*1024 ! dimension of x and y arrays
  integer             ::  i,j,iterk,istat
  
  real(dp), allocatable,target  :: A(:, :), Anew(:, :)
  real(dp), pointer             :: A_pointer1(:,:)
  real(dp), pointer             :: A_pointer2(:,:)
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

