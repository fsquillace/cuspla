#pragma once

// CULA includes
#include <culapack.h>
#include <culapackdevice.h>
#include <cula.h>
#include <culablas.h>
#include <culablasdevice.h>
#include <culastatus.h>


#include <cusp/memory.h>
#include <cusp/array2d.h>


// THRUST includes
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>


#include "cusplautils.h"


namespace cuspla{

// *****************  Eigenvalue problem *****************************
template <typename Array2d, typename Array1d>
culaStatus geev(Array2d& H, Array1d& eigvals, Array2d& eigvects, float, cusp::host_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    cusp::array1d<ValueType, MemorySpace> eigvals_im(H.num_rows);


	status = culaSgeev('N', 'V', H.num_cols, thrust::raw_pointer_cast(H.values.data()),
					   H.num_rows,
					   thrust::raw_pointer_cast(eigvals.data()),
					   thrust::raw_pointer_cast(eigvals_im.data()),
					   NULL, 1,
					   thrust::raw_pointer_cast(eigvects.values.data()), H.num_rows);
    return status;
}

template <typename Array2d, typename Array1d>
culaStatus geev(Array2d& H, Array1d& eigvals, Array2d& eigvects, double, cusp::host_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    cusp::array1d<ValueType, MemorySpace> eigvals_im(H.num_rows);


	status = culaDgeev('N', 'V', H.num_cols, thrust::raw_pointer_cast(H.values.data()),
					   H.num_cols,
					   thrust::raw_pointer_cast(eigvals.data()),
					   thrust::raw_pointer_cast(eigvals_im.data()),
					   NULL, 1,
					   thrust::raw_pointer_cast(eigvects.values.data()), H.num_rows);
    return status;
}

template <typename Array2d, typename Array1d>
culaStatus geev(Array2d& H, Array1d& eigvals, Array2d& eigvects, float, cusp::device_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    cusp::array1d<ValueType, MemorySpace> eigvals_im(H.num_rows);

	status = culaDeviceSgeev('N', 'V', H.num_cols, thrust::raw_pointer_cast(H.values.data()),
					   H.num_cols,
					   thrust::raw_pointer_cast(eigvals.data()),
					   thrust::raw_pointer_cast(eigvals_im.data()),
					   NULL, 1,
					   thrust::raw_pointer_cast(eigvects.values.data()), H.num_rows);
    return status;
}

template <typename Array2d, typename Array1d>
culaStatus geev(Array2d& H, Array1d& eigvals, Array2d& eigvects, double, cusp::device_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    cusp::array1d<ValueType, MemorySpace> eigvals_im(H.num_rows);


	status = culaDeviceDgeev('N', 'V', H.num_cols, thrust::raw_pointer_cast(H.values.data()),
					   H.num_cols,
					   thrust::raw_pointer_cast(eigvals.data()),
					   thrust::raw_pointer_cast(eigvals_im.data()),
					   NULL, 1,
					   thrust::raw_pointer_cast(eigvects.values.data()), H.num_rows);
    return status;
}


// ------------------   Entry point ---------------------
template <typename Array2d, typename Array1d>
culaStatus geev(Array2d& H, Array1d& eigvals, Array2d& eigvects){

	eigvals.resize(H.num_rows);
	eigvects.resize(H.num_rows, H.num_rows);
    return geev(H, eigvals, eigvects, typename Array2d::value_type(), typename Array2d::memory_space(), typename Array2d::orientation());
}




// ***************** Matrix-Matrix multiplication ******************

template <typename ValueType, typename Array2d>
culaStatus gemm(Array2d& A, Array2d& B, Array2d& C, char tA, char tB, ValueType alpha,\
        size_t n, size_t m, size_t k, size_t lda, size_t ldb, \
        float, cusp::host_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
//    typedef typename Array2d::value_type   ValueType;

	status = culaSgemm(tA,tB, n, m, k, alpha,
			thrust::raw_pointer_cast(A.values.data()),
			lda,
			thrust::raw_pointer_cast(B.values.data()),
			ldb,
			ValueType(0),
			thrust::raw_pointer_cast(C.values.data()),
			n);


    return status;
}

template <typename ValueType, typename Array2d>
culaStatus gemm(Array2d& A, Array2d& B, Array2d& C, char tA, char tB, ValueType alpha,\
        size_t n, size_t m, size_t k, size_t lda, size_t ldb, \
        float, cusp::device_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
//    typedef typename Array2d::value_type   ValueType;

	status = culaDeviceSgemm(tA,tB, n, m, k, alpha,
            thrust::raw_pointer_cast(A.values.data()),
            lda,
            thrust::raw_pointer_cast(B.values.data()),
            ldb,
            ValueType(0),
            thrust::raw_pointer_cast(C.values.data()),
            n);


    return status;
}

template <typename ValueType, typename Array2d>
culaStatus gemm(Array2d& A, Array2d& B, Array2d& C, char tA, char tB, ValueType alpha,\
        size_t n, size_t m, size_t k, size_t lda, size_t ldb, \
        double, cusp::host_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
//    typedef typename Array2d::value_type   ValueType;

	status = culaDgemm(tA,tB, n, m, k, alpha,
            thrust::raw_pointer_cast(A.values.data()),
            lda,
            thrust::raw_pointer_cast(B.values.data()),
            ldb,
            ValueType(0),
            thrust::raw_pointer_cast(C.values.data()),
            n);


    return status;
}

template <typename ValueType, typename Array2d>
culaStatus gemm(Array2d& A, Array2d& B, Array2d& C, char tA, char tB, ValueType alpha,\
        size_t n, size_t m, size_t k, size_t lda, size_t ldb, \
        double, cusp::device_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
//    typedef typename Array2d::value_type   ValueType;

	status = culaDeviceDgemm(tA,tB, n, m, k, alpha,
            thrust::raw_pointer_cast(A.values.data()),
            lda,
            thrust::raw_pointer_cast(B.values.data()),
            ldb,
            ValueType(0),
            thrust::raw_pointer_cast(C.values.data()),
            n);


    return status;
}

// --------- Entry point -----------------------
// The upper triangular matrix R will be in H matrix
template <typename Array2d, typename ValueType>
culaStatus gemm(Array2d& A, Array2d& B, Array2d& C, ValueType alpha=ValueType(1),\
		ValueType beta = ValueType(0),  bool transA=false, bool transB=false){
    /*
     * C = alpha*OP(A)*OP(B) + beta*C
     *
     * Default values:
     *   alpha = 1.0
     *   beta = 0.0
     *   transA = transB = false
     *
     * Note: C must be different from both A and B matrix because storing
     *  the result in C will change either matrix A or B going
     *  to change the correct result.
     */


    assert(&A!=&C && &B!=&C);

	char tA = (transA)?'T':'N';
	char tB = (transB)?'T':'N';


    size_t n = A.num_rows;
    size_t kA = A.num_cols;
    size_t lda = n;
    if(transA){
        n = A.num_cols;
        kA = A.num_rows;
        lda = kA;
    }

    size_t m = B.num_cols;
    size_t kB = B.num_rows;
    size_t ldb = kB;
    if(transB){
        m = B.num_rows;
        kB = B.num_cols;
        ldb = m;
    }

    assert(kA == kB);

	C.resize(n,m);
	return gemm(A, B, C, tA, tB, alpha, n, m, kA, lda, ldb,   \
	        typename Array2d::value_type(),   \
	        typename Array2d::memory_space(), \
	        typename Array2d::orientation());
}




// ***************** Matrix-Vector multiplication ******************

template <typename Array2d, typename Array1d>
culaStatus gemv(Array2d& A, Array1d& x, Array1d& y, char tA,\
        float, cusp::host_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

	status = culaSgemv(tA, A.num_rows, A.num_cols, ValueType(1),
			thrust::raw_pointer_cast(A.values.data()),
			A.num_rows,
			thrust::raw_pointer_cast(x.data()),
			ValueType(1),
			ValueType(0),
			thrust::raw_pointer_cast(y.data()),
			ValueType(1));


    return status;
}

template <typename Array2d, typename Array1d>
culaStatus gemv(Array2d& A, Array1d& x, Array1d& y, char tA,\
        float, cusp::device_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

	status = culaDeviceSgemv(tA, A.num_rows, A.num_cols, ValueType(1),
			thrust::raw_pointer_cast(A.values.data()),
			A.num_rows,
			thrust::raw_pointer_cast(x.data()),
			ValueType(1),
			ValueType(0),
			thrust::raw_pointer_cast(y.data()),
			ValueType(1));


    return status;
}

template <typename Array2d, typename Array1d>
culaStatus gemv(Array2d& A, Array1d& x, Array1d& y, char tA,\
        double, cusp::host_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

	status = culaDgemv(tA, A.num_rows, A.num_cols, ValueType(1),
			thrust::raw_pointer_cast(A.values.data()),
			A.num_rows,
			thrust::raw_pointer_cast(x.data()),
			ValueType(1),
			ValueType(0),
			thrust::raw_pointer_cast(y.data()),
			ValueType(1));


    return status;
}

template <typename Array2d, typename Array1d>
culaStatus gemv(Array2d& A, Array1d& x, Array1d& y, char tA,\
        double, cusp::device_memory, cusp::column_major){
    culaStatus status;

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

	status = culaDeviceDgemv(tA, A.num_rows, A.num_cols,ValueType(1),
			thrust::raw_pointer_cast(A.values.data()),
			A.num_rows,
			thrust::raw_pointer_cast(x.data()),
			ValueType(1),
			ValueType(0),
			thrust::raw_pointer_cast(y.data()),
			ValueType(1));


    return status;
}



// ---------------------- Entry point --------------------------
template <typename Array2d, typename Array1d>
culaStatus gemv(Array2d& A, Array1d& x, Array1d& y, bool transA=false){
    /*
    *  y must be different from x.
    */

    assert(&x != &y);
	char tA = (transA)?'T':'N';

    size_t n = A.num_rows;
    size_t m = A.num_cols;
    if(transA){
        n = A.num_cols;
        m = A.num_rows;
    }

    assert(m == x.size());
    y.resize(n);
	return gemv(A, x, y, tA, typename Array2d::value_type(), typename Array2d::memory_space(), typename Array2d::orientation());
}



// *****************  QR Factorization *****************************


// Computes H = I-tau*v*v'
template<typename ValueType, typename Array1d, typename Array2d>
void house_holder(const ValueType tau, const Array1d& v, Array2d& H){

    typedef typename Array2d::value_type   ValueType2;

    size_t N = v.size();
	H.resize(N, N);

    Array2d tmp(N, 1);
    tmp.values = v;
//    thrust::copy(v.begin(), v.end(), tmp.values.begin());

    gemm(tmp, tmp, H, -tau, ValueType2(0), false, true);

    // Adds 1 of each element of the diagonal
    thrust::counting_iterator<int> stencil (0);
    thrust::transform_if(H.values.begin(), H.values.end(), \
            stencil, \
            H.values.begin(), \
            cuspla::plus_const<ValueType2>(ValueType2(1)), \
            cuspla::in_diagonal(N,N));

}




template <typename Array2d, typename Array1d>
culaStatus geqrf(Array2d& A, Array1d& tau, float, cusp::host_memory, cusp::column_major){
	return culaSgeqrf(A.num_rows, A.num_cols, thrust::raw_pointer_cast(A.values.data()),
					   A.num_rows,
					   thrust::raw_pointer_cast(tau.data()));
}

template <typename Array2d, typename Array1d>
culaStatus geqrf(Array2d& A, Array1d& tau, double, cusp::host_memory, cusp::column_major){
	return culaDgeqrf(A.num_rows, A.num_cols, thrust::raw_pointer_cast(A.values.data()),
					   A.num_rows,
					   thrust::raw_pointer_cast(tau.data()));
}

template <typename Array2d, typename Array1d>
culaStatus geqrf(Array2d& A, Array1d& tau, float, cusp::device_memory, cusp::column_major){
	return culaDeviceSgeqrf(A.num_rows, A.num_cols, thrust::raw_pointer_cast(A.values.data()),
					   A.num_rows,
					   thrust::raw_pointer_cast(tau.data()));
}

template <typename Array2d, typename Array1d>
culaStatus geqrf(Array2d& A, Array1d& tau, double, cusp::device_memory, cusp::column_major){
	return culaDeviceDgeqrf(A.num_rows, A.num_cols, thrust::raw_pointer_cast(A.values.data()),
					   A.num_rows,
					   thrust::raw_pointer_cast(tau.data()));
}







// ------------------ Entry point ----------------
// The upper triangular matrix R will be in H matrix
template <typename Array2d>
culaStatus geqrf(Array2d& A, Array2d& Q, Array2d& R, bool get_R=true){


    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    size_t N = A.num_rows;
    size_t M = A.num_cols;
    size_t min_dim = std::min(M,N);

    cusp::array1d<ValueType, MemorySpace> tau(min_dim);

	culaStatus status = geqrf(A, tau, typename Array2d::value_type(), \
	        typename Array2d::memory_space(), \
	        typename Array2d::orientation());


	cusp::array1d<ValueType, MemorySpace> v(N, ValueType(0));
	cusp::array2d<ValueType, MemorySpace, cusp::column_major> H(N,M);

	// Set Q to the identity
	Q.resize(N,N);
	thrust::fill(Q.values.begin(), Q.values.end(), ValueType(0));
    //     Complete the diagonal
    thrust::counting_iterator<int> stencil (0);
    thrust::transform_if(Q.values.begin(), Q.values.end(), \
        stencil, \
        Q.values.begin(), \
        cuspla::assigns<ValueType>(ValueType(1)), \
        cuspla::in_diagonal(N,N));


	// Computes Q = Q*H(k)
	for(size_t k = 0; k<min_dim; k++){
		// define v
		thrust::fill(v.begin(), v.begin()+k, ValueType(0));
		v[k]=ValueType(1);
		thrust::copy(A.values.begin()+(N*k + k+1), A.values.begin()+(N*(k+1)), v.begin()+k+1);



		house_holder(tau[k], v, H);


        Array2d Q_;
		gemm(Q, H, Q_, ValueType(1));
		cusp::copy(Q_, Q);

	}


    if(get_R){
        //computes R is the upper triangular of A
        R.resize(N,M);
        thrust::fill(R.values.begin(), R.values.end(), ValueType(0));
        // Copy the upper triangular of A to R
        thrust::transform_if(A.values.begin(), A.values.end(), \
            thrust::counting_iterator<int>(0), \
            R.values.begin(), \
            cuspla::copy<ValueType>(), \
            cuspla::in_upper_triang(N,M));

    }

	return status;

}




} // end cula namespace

