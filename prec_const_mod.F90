MODULE prec_const
  use, intrinsic :: iso_fortran_env, only: REAL32, REAL64, &
                                           stdin=>input_unit, &
                                           stdout=>output_unit, &
                                           stderr=>error_unit
  !
  !   Precision for real and complex
  !
  INTEGER, PARAMETER :: sp = REAL32 !Single precision, should not be used
  INTEGER, PARAMETER :: dp = REAL64 !real(dp), enforced through the code

  INTEGER, private :: dp_r, dp_p !Range and Aprecision of doubles
  INTEGER, private :: sp_r, sp_p !Range and precision of singles


  INTEGER, private :: MPI_SP !Single precision for MPI
  INTEGER, private :: MPI_DP !Double precision in MPI

  REAL(dp),public :: nan_
  !
  !   Some useful constants
  !
  REAL(dp), PARAMETER :: PI=3.141592653589793238462643383279502884197_dp
  REAL(dp), PARAMETER :: PIO2=1.57079632679489661923132169163975144209858_dp
  REAL(dp), PARAMETER :: TWOPI=6.283185307179586476925286766559005768394_dp
  REAL(dp), PARAMETER :: SQRT2=1.41421356237309504880168872420969807856967_dp
  !
  CONTAINS
    SUBROUTINE INIT_PREC_CONST
      IMPLICIT NONE
      integer :: ierr,me
      
      REAL(sp) :: a = 1_sp
      REAL(dp) :: b = 1_dp
!      logical :: commute = .true.

      !Get range and precision of ISO FORTRAN sizes
      sp_r = range(a)
      sp_p = precision(a)

      dp_r = range(b)
      dp_p = precision(b)

      nan_ = 0._dp/1._dp
    END SUBROUTINE INIT_PREC_CONST



END MODULE prec_const
