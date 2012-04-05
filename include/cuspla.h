
#include <culastatus.h>



namespace cuspla{

template <typename Array2d, typename Array1d>
culaStatus geev(Array2d& H, Array1d& eigvals, Array2d& eigvects);

template <typename Array2d, typename ValueType>
culaStatus gemm(Array2d& A, Array2d& B, Array2d& C, ValueType alpha=ValueType(1),\
		ValueType beta = ValueType(0),  bool transA=false, bool transB=false);

template <typename Array2d, typename Array1d>
culaStatus gemv(Array2d& A, Array1d& x, Array1d& y, bool transA=false);

template <typename Array2d>
culaStatus geqrf(Array2d& A, Array2d& Q, Array2d& R, bool get_R=true);


template <typename Array2d>
culaStatus getri(Array2d& A);

}


