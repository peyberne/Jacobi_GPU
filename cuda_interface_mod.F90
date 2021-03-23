MODULE cuda_interface

use prec_const
INTERFACE
  FUNCTION allocate_cuda_memory(n) BIND(C, NAME='allocate_cuda_memory')
    use iso_c_binding
    TYPE(C_PTR) :: allocate_cuda_memory
    integer :: n
  END FUNCTION allocate_cuda_memory


  subroutine allocate_cuda_struct(cudastruct)  bind(C, name='allocate_cuda_struct') 
     use iso_c_binding
     type(c_ptr):: cudastruct
  end subroutine allocate_cuda_struct

  subroutine fill_cuda_struct(cudastruct,izeg,izsg,iyeg,iysg,ixeg,ixsg)  bind(C, name='fill_cuda_struct') 
  !subroutine fill_cuda_struct(cudastruct)  bind(C, name='fill_cuda_struct') 
     use iso_c_binding
     type(c_ptr):: cudastruct
     integer :: izeg,izsg,iyeg,iysg,ixeg,ixsg
  end subroutine fill_cuda_struct

END INTERFACE
  INTERFACE gbs_allocate_cuda
     MODULE PROCEDURE gbs_allocate_cuda_dp3, gbs_allocate_cuda_dp2
  END INTERFACE gbs_allocate_cuda
  real(dp), pointer, save :: strmf_cpu(:,:,:),strmf_gpu(:,:,:), indata(:,:,:)
  real(dp), pointer, save :: strmf1_cpu(:,:,:), strmf1_gpu(:,:,:)
  real(dp), pointer, save :: grady_f(:,:,:), grady_diff(:,:,:), diff(:,:,:)
  real(dp), pointer, save :: f_z(:,:,:), f_y(:,:,:), f_n(:,:,:), f_x_n(:,:,:)
  real(dp), pointer, save :: gradpar_x_n_gpu(:,:), gradpar_x_v_gpu(:,:), gradpar_y_n_gpu(:,:), gradpar_y_v_gpu(:,:)
  !real(dp), pointer, save :: gradpar_x_n_gpu(:,:,:), gradpar_x_v_gpu(:,:,:), gradpar_y_n_gpu(:,:,:), gradpar_y_v_gpu(:,:,:)
contains

  SUBROUTINE gbs_allocate_cuda_dp3(a,is1,ie1,is2,ie2,is3,ie3)
    use iso_c_binding
    real(dp), DIMENSION(:,:,:), pointer, INTENT(INOUT) :: a
    INTEGER, INTENT(IN) :: is1,ie1,is2,ie2,is3,ie3
    integer ndata
    ndata=(ie3-is3+1)*(ie2-is2+1)*(ie1-is1+1) 
    call c_f_pointer(allocate_cuda_memory(ndata), a, [(ie1-is1+1),(ie2-is2+1),(ie3-is3+1)])
    a(is1:,is2:,is3:) => a
    a=0.0_dp
  END SUBROUTINE gbs_allocate_cuda_dp3

  SUBROUTINE gbs_allocate_cuda_dp2(a,is1,ie1,is2,ie2)
    use iso_c_binding
    real(dp), DIMENSION(:,:), pointer, INTENT(INOUT) :: a
    INTEGER, INTENT(IN) :: is1,ie1,is2,ie2
    integer ndata

    ndata=(ie2-is2+1)*(ie1-is1+1) 
    call c_f_pointer(allocate_cuda_memory(ndata), a, [(ie1-is1+1),(ie2-is2+1)])
    a(is1:,is2:) => a
    a=0.0_dp
  END SUBROUTINE gbs_allocate_cuda_dp2


END MODULE cuda_interface
