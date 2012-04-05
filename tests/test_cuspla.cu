/*
 * test_cuspla.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: Squillace Filippo
 */

//#define CUSP_USE_TEXTURE_MEMORY


// CUSP includes
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/detail/random.h>


#include <cuspla.cu>


// common
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestSuite.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

#include <string.h>
#include <iostream>
#include <stdio.h>
#include <sstream>

void checkStatus(culaStatus status)
{
    if(!status)
        return;
    if(status == culaArgumentError)
        printf("Invalid value for parameter %d\n", culaGetErrorInfo());
    else if(status == culaDataError)
        printf("Data error (%d)\n", culaGetErrorInfo());
    else if(status == culaBlasError)
        printf("Blas error (%d)\n", culaGetErrorInfo());
    else if(status == culaRuntimeError)
        printf("Runtime error (%d)\n", culaGetErrorInfo());
    else
        printf("%s\n", culaGetStatusString(status));

    culaShutdown();
    exit(EXIT_FAILURE);
}



class CusplaTestCase : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE (CusplaTestCase);

    CPPUNIT_TEST (test_host_GEMM);
    CPPUNIT_TEST (test_host_transpose_GEMM);
    CPPUNIT_TEST (test_device_GEMM);
    CPPUNIT_TEST (test_device_transpose_GEMM);

    CPPUNIT_TEST (test_host_GEMV);
    CPPUNIT_TEST(test_host_transpose_GEMV);
    CPPUNIT_TEST(test_device_GEMV);
    CPPUNIT_TEST(test_device_traspose_GEMV);

    CPPUNIT_TEST(test_host_GEQRF);
    CPPUNIT_TEST(test_device_GEQRF);

    CPPUNIT_TEST(test_host_GEEV);
    CPPUNIT_TEST(test_device_GEEV);

    CPPUNIT_TEST(test_host_GETRI);
    CPPUNIT_TEST(test_device_GETRI);

    CPPUNIT_TEST_SUITE_END ();

    typedef int    IndexType;
    typedef float ValueType;
    typedef cusp::array2d<ValueType,cusp::device_memory, cusp::column_major> DeviceMatrix_array2d;
    typedef cusp::array2d<ValueType, cusp::host_memory, cusp::column_major>   HostMatrix_array2d;
    typedef cusp::array1d<ValueType,cusp::device_memory> DeviceVector_array1d;
    typedef cusp::array1d<ValueType, cusp::host_memory>   HostVector_array1d;
private:

    std::vector<std::string> path_not_squared;
    std::vector<DeviceMatrix_array2d> dev_mat_not_squared;
    std::vector<HostMatrix_array2d> host_mat_not_squared;


    std::vector<std::string> path_def_pos;
    std::vector<DeviceMatrix_array2d> dev_mat_def_pos;
    std::vector<HostMatrix_array2d> host_mat_def_pos;

public:

  void setUp()
  {

      culaStatus status;
      status = culaInitialize();
      checkStatus(status);


      // ################################ NOT SQUARED #####################
//      path_not_squared = std::vector<std::string>(1);
//      path_not_squared[0] = "data/not-squared/rand90x80.mtx";
      path_not_squared = std::vector<std::string>(5);
      path_not_squared[0] = "data/not-squared/rand9x11.mtx";
      path_not_squared[1] = "data/not-squared/rand10x9.mtx";
      path_not_squared[2] = "data/not-squared/rand90x80.mtx";
      path_not_squared[3] = "data/not-squared/rand90x100.mtx";
      path_not_squared[4] = "data/not-squared/rand100x90.mtx";

      host_mat_not_squared = std::vector<HostMatrix_array2d>(path_not_squared.size());
      dev_mat_not_squared = std::vector<DeviceMatrix_array2d>(path_not_squared.size());
      for(size_t i=0; i<path_not_squared.size(); i++){
          cusp::io::read_matrix_market_file(host_mat_not_squared[i], path_not_squared[i]);
          dev_mat_not_squared[i] = DeviceMatrix_array2d(host_mat_not_squared[i]);
      }


      // ################################ POSITIVE DEFINITE #####################
      path_def_pos = std::vector<std::string>(3);
      path_def_pos[0] = "data/positive-definite/lehmer10.mtx";
      path_def_pos[1] = "data/positive-definite/lehmer50.mtx";
      path_def_pos[2] = "data/positive-definite/lehmer100.mtx";

      host_mat_def_pos = std::vector<HostMatrix_array2d>(path_def_pos.size());
      dev_mat_def_pos = std::vector<DeviceMatrix_array2d>(path_def_pos.size());
      for(size_t i=0; i<path_def_pos.size(); i++){
          cusp::io::read_matrix_market_file(host_mat_def_pos[i], path_def_pos[i]);
          dev_mat_def_pos[i] = DeviceMatrix_array2d(host_mat_def_pos[i]);
      }


  }

  void tearDown()
  {
      culaShutdown();
  }


  void test_host_GEMM()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){
          for(size_t j=0; j<path_not_squared.size(); j++){

              if(host_mat_not_squared[i].num_cols != host_mat_not_squared[j].num_rows)
                  continue;

              HostMatrix_array2d C;
              HostMatrix_array2d C2;

              cuspla::gemm(host_mat_not_squared[i], host_mat_not_squared[j], C,\
            		  ValueType(1),ValueType(0),false,false);
              cusp::multiply(host_mat_not_squared[i], host_mat_not_squared[j], C2);

              ValueType errRel = nrmVector("host_GEMM "+path_not_squared[i]+" "+path_not_squared[j], C.values, C2.values);
              CPPUNIT_ASSERT( errRel < 1.0e-6 );
          }
      }
  }


  void test_host_transpose_GEMM()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){
          for(size_t j=0; j<path_not_squared.size(); j++){

              if(host_mat_not_squared[i].num_rows != host_mat_not_squared[j].num_cols)
                   continue;

              HostMatrix_array2d C;
              HostMatrix_array2d C2, mat_OP1_trans, mat_OP2_trans;

              cuspla::gemm(host_mat_not_squared[i], host_mat_not_squared[j], C,\
            		  ValueType(1),ValueType(0),true, true);

              cusp::transpose(host_mat_not_squared[i], mat_OP1_trans);
              cusp::transpose(host_mat_not_squared[j], mat_OP2_trans);
              cusp::multiply(mat_OP1_trans, mat_OP2_trans, C2);

              ValueType errRel = nrmVector("host_transpose_GEMM "+path_not_squared[i]+" "+path_not_squared[j], C.values, C2.values);
              CPPUNIT_ASSERT( errRel < 1.0e-6 );
          }
      }
  }


  void test_device_GEMM()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){
          for(size_t j=0; j<path_not_squared.size(); j++){

              if(dev_mat_not_squared[i].num_cols != dev_mat_not_squared[j].num_rows)
                                continue;

              DeviceMatrix_array2d C;
              HostMatrix_array2d C2, C_host;

              cuspla::gemm(dev_mat_not_squared[i], dev_mat_not_squared[j], C,\
            		  ValueType(1),ValueType(0),false,false);
              C_host = HostMatrix_array2d(C);

              cusp::multiply(host_mat_not_squared[i], host_mat_not_squared[j], C2);

              ValueType errRel = nrmVector("device_GEMM "+path_not_squared[i]+" "+path_not_squared[j], C_host.values, C2.values);
              CPPUNIT_ASSERT( errRel < 1.0e-6 );
          }
      }
  }


  void test_device_transpose_GEMM()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){
          for(size_t j=0; j<path_not_squared.size(); j++){

              if(dev_mat_not_squared[i].num_rows != dev_mat_not_squared[j].num_cols)
                                 continue;

              DeviceMatrix_array2d C;
              HostMatrix_array2d C2, mat_OP1_trans, mat_OP2_trans, C_host;

              cuspla::gemm(dev_mat_not_squared[i], dev_mat_not_squared[j], C,\
            		  ValueType(1),ValueType(0),true, true);
              C_host = HostMatrix_array2d(C);

              cusp::transpose(host_mat_not_squared[i], mat_OP1_trans);
              cusp::transpose(host_mat_not_squared[j], mat_OP2_trans);
              cusp::multiply(mat_OP1_trans, mat_OP2_trans, C2);

              ValueType errRel = nrmVector("device_transpose_GEMM "+path_not_squared[i]+" "+path_not_squared[j], C_host.values, C2.values);
              CPPUNIT_ASSERT( errRel < 1.0e-6 );

          }
      }
  }


  void test_host_GEMV()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){
          size_t N = host_mat_not_squared[i].num_rows;
          size_t M = host_mat_not_squared[i].num_cols;
          HostVector_array1d vec_OP2 = cusp::detail::random_reals<ValueType>(M);
          HostVector_array1d C(N);
          HostVector_array1d C2(N);

          cuspla::gemv(host_mat_not_squared[i], vec_OP2, C,false);

          cusp::multiply(host_mat_not_squared[i], vec_OP2, C2);

          ValueType errRel = nrmVector("host_GEMV "+path_not_squared[i], C, C2);
          CPPUNIT_ASSERT( errRel < 1.0e-6 );
      }
  }

  void test_host_transpose_GEMV()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){
          size_t N = host_mat_not_squared[i].num_rows;
          size_t M = host_mat_not_squared[i].num_cols;
          HostVector_array1d vec_OP2 = cusp::detail::random_reals<ValueType>(N);
          HostVector_array1d C(M);
          HostVector_array1d C2(M);
          HostMatrix_array2d mat_OP1_trans;

          cuspla::gemv(host_mat_not_squared[i], vec_OP2, C,true);

          cusp::transpose(host_mat_not_squared[i], mat_OP1_trans);
          cusp::multiply(mat_OP1_trans, vec_OP2, C2);

          ValueType errRel = nrmVector("host_transpose_GEMV "+path_not_squared[i], C, C2);
          CPPUNIT_ASSERT( errRel < 1.0e-6 );
    }
  }

  void test_device_GEMV()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){

          size_t N = host_mat_not_squared[i].num_rows;
          size_t M = host_mat_not_squared[i].num_cols;
          DeviceVector_array1d vec_OP2 = cusp::detail::random_reals<ValueType>(M);
          DeviceVector_array1d C(N);
          HostVector_array1d C2(N), vec_OP2_host, C_host;

          cuspla::gemv(dev_mat_not_squared[i], vec_OP2, C,false);

          vec_OP2_host = HostVector_array1d(vec_OP2);
          cusp::multiply(host_mat_not_squared[i], vec_OP2_host, C2);
          C_host = HostVector_array1d(C);

          ValueType errRel = nrmVector("device_GEMV "+path_not_squared[i], C_host, C2);

          CPPUNIT_ASSERT( errRel < 1.0e-6 );

      }
  }

  void test_device_traspose_GEMV()
  {
      for(size_t i=0; i<path_not_squared.size(); i++){

          size_t N = host_mat_not_squared[i].num_rows;
          size_t M = host_mat_not_squared[i].num_cols;
          DeviceVector_array1d vec_OP2 = cusp::detail::random_reals<ValueType>(N);
          DeviceVector_array1d C(M);
          HostVector_array1d C2(M), vec_OP2_host, C_host;

          cuspla::gemv(dev_mat_not_squared[i], vec_OP2, C,true);

          vec_OP2_host = HostVector_array1d(vec_OP2);
          HostMatrix_array2d mat_OP1_trans;
          cusp::transpose(host_mat_not_squared[i], mat_OP1_trans);
          cusp::multiply(mat_OP1_trans, vec_OP2_host, C2);
          C_host = HostVector_array1d(C);


          ValueType errRel = nrmVector("device_traspose_GEMV "+path_not_squared[i], C_host, C2);
          CPPUNIT_ASSERT( errRel < 1.0e-6 );

      }
  }


  void test_host_GEQRF()
  {


      for(size_t i=0; i<path_not_squared.size(); i++){

          size_t n = host_mat_not_squared[i].num_rows;
          size_t m = host_mat_not_squared[i].num_cols;
          HostMatrix_array2d Q(n,n);
          HostMatrix_array2d R(n, m);
          HostMatrix_array2d A(n,m), mat_OP1_copy;

          cusp::copy(host_mat_not_squared[i], mat_OP1_copy);
          cuspla::geqrf(mat_OP1_copy, Q, R, true);

          //Checks orthogonality of Q
          HostMatrix_array2d Qt(n,n), I(n,n), Ip(n,n);
          cusp::transpose(Q, Qt);
          cusp::multiply(Q,Qt, I);
          thrust::fill(Ip.values.begin(), Ip.values.end(), ValueType(0));
          thrust::counting_iterator<int> stencil (0);
          thrust::transform_if(Ip.values.begin(), Ip.values.end(), \
              stencil, \
              Ip.values.begin(), \
              cuspla::assigns<ValueType>(ValueType(1)), \
              cuspla::in_diagonal(n,n));
          ValueType errRel = nrmVector("host_GEQRF orthogonality "+path_not_squared[i], I.values, Ip.values);
          CPPUNIT_ASSERT( errRel < 1.0e-5 );



          // Checks Factorization
          cusp::multiply(Q, R, A);
          errRel = nrmVector("host_GEQRF factorization "+path_not_squared[i], host_mat_not_squared[i].values, A.values);
          CPPUNIT_ASSERT( errRel < 1.0e-5 );

      }
  }

  void test_device_GEQRF()
  {

      for(size_t i=0; i<path_not_squared.size(); i++){
          size_t n = host_mat_not_squared[i].num_rows;
          size_t m = host_mat_not_squared[i].num_cols;
          DeviceMatrix_array2d Q(n,n);
          DeviceMatrix_array2d R(n, m);


          cuspla::geqrf(dev_mat_not_squared[i], Q, R, true);

          //Checks orthogonality of Q
          HostMatrix_array2d I(n,n), Ip(n,n);
          HostMatrix_array2d Q_host,Qt(n,n);
          cusp::copy(Q, Q_host);
          cusp::transpose(Q_host, Qt);
          cusp::multiply(Q_host,Qt, I);
          thrust::fill(Ip.values.begin(), Ip.values.end(), ValueType(0));
          thrust::counting_iterator<int> stencil (0);
          thrust::transform_if(Ip.values.begin(), Ip.values.end(), \
              stencil, \
              Ip.values.begin(), \
              cuspla::assigns<ValueType>(ValueType(1)), \
              cuspla::in_diagonal(n,n));
          ValueType errRel = nrmVector("device_GEQRF orthogonality "+path_not_squared[i], I.values, Ip.values);
          CPPUNIT_ASSERT( errRel < 1.0e-5 );



          // Checks Factorization
          HostMatrix_array2d A(n,m), R_host(n, m);
          cusp::copy(R, R_host);
          cusp::multiply(Q_host, R_host, A);
          errRel = nrmVector("device_GEQRF factorization "+path_not_squared[i], host_mat_not_squared[i].values, A.values);
          CPPUNIT_ASSERT( errRel < 1.0e-5 );

      }
  }



  void test_host_GEEV()
  {

      for(size_t i=0; i<path_def_pos.size(); i++){

          size_t n = host_mat_def_pos[i].num_rows;
          size_t m = host_mat_def_pos[i].num_cols;
          HostMatrix_array2d eigvects;
          HostVector_array1d eigvals;
          HostMatrix_array2d mat_OP1_copy;
          HostVector_array1d y1, eigvec(m);

          cusp::copy(host_mat_def_pos[i], mat_OP1_copy);
          cuspla::geev(mat_OP1_copy, eigvals, eigvects);

          for(size_t j=0; j<eigvals.size(); j++){
              thrust::copy(eigvects.values.begin()+ j*m, eigvects.values.begin()+ (j+1)*m,eigvec.begin());
              cuspla::gemv(host_mat_def_pos[i], eigvec, y1, false);
              cusp::blas::scal(eigvec, (ValueType)eigvals[j]);

              std::stringstream j_str, eigval_str;
              j_str << j;
              eigval_str << eigvals[j];

              ValueType errRel = nrmVector("host_GEEV eigval["+j_str.str()+"]:"+eigval_str.str()+" "+path_def_pos[i], y1, eigvec);
              CPPUNIT_ASSERT( errRel < 1.0e-2 );

          }
      }
  }

  void test_device_GEEV()
  {

      for(size_t i=0; i<path_def_pos.size(); i++){

          size_t n = host_mat_def_pos[i].num_rows;
          size_t m = host_mat_def_pos[i].num_cols;
          DeviceMatrix_array2d eigvects;
          DeviceVector_array1d eigvals;
          DeviceMatrix_array2d mat_OP1_copy;
          HostVector_array1d y1, eigvec(m);

          cusp::copy(dev_mat_def_pos[i], mat_OP1_copy);
          cuspla::geev(mat_OP1_copy, eigvals, eigvects);

          for(size_t j=0; j<eigvals.size(); j++){
              thrust::copy(eigvects.values.begin()+ j*m, eigvects.values.begin()+ (j+1)*m,eigvec.begin());
              cuspla::gemv(host_mat_def_pos[i], eigvec, y1, false);
              cusp::blas::scal(eigvec, (ValueType)eigvals[j]);

              std::stringstream j_str, eigval_str;
              j_str << j;
              eigval_str << eigvals[j];

              ValueType errRel = nrmVector("device_GEEV eigval["+j_str.str()+"]:"+eigval_str.str()+" "+path_def_pos[i], y1, eigvec);
              CPPUNIT_ASSERT( errRel < 1.0e-2 );

          }
      }
  }


  void test_host_GETRI()
  {

      for(size_t i=0; i<path_def_pos.size(); i++){
    	  HostMatrix_array2d A_inv;
    	  cusp::copy(host_mat_def_pos[i], A_inv);
    	  cuspla::getri(A_inv);
    	  cuspla::getri(A_inv);

          ValueType errRel = nrmVector("host_GETRIEV "+path_def_pos[i], A_inv.values, host_mat_def_pos[i].values);
          CPPUNIT_ASSERT( errRel < 1.0e-2 );

      }
  }


  void test_device_GETRI()
  {

      for(size_t i=0; i<path_def_pos.size(); i++){
    	  DeviceMatrix_array2d A_inv;
    	  cusp::copy(dev_mat_def_pos[i], A_inv);
    	  cuspla::getri(A_inv);
    	  cuspla::getri(A_inv);

    	  HostMatrix_array2d A_inv_host;
    	  cusp::copy(A_inv, A_inv_host);
          ValueType errRel = nrmVector("host_GETRIEV "+path_def_pos[i], A_inv_host.values, host_mat_def_pos[i].values);
          CPPUNIT_ASSERT( errRel < 1.0e-2 );

      }
  }




template <typename Array1d>
ValueType nrmVector(std::string title, Array1d& A, Array1d& A2){
      ValueType nrmA = cusp::blas::nrm2(A);
      ValueType nrmA2 = cusp::blas::nrm2(A2);
      // Calculates the difference and overwrite the matrix C
      cusp::blas::axpy(A, A2, ValueType(-1));
      ValueType nrmDiff = cusp::blas::nrm2(A2);



      ValueType errRel = ValueType(0);
      if(nrmA==ValueType(0))
          errRel = ValueType(1.0e-30);
      else
          errRel = nrmDiff/nrmA;

#ifdef VERBOSE
#ifndef VVERBOSE
      if(errRel != errRel || errRel >= 1.0e-2){ // Checks if error is nan
#endif VVERBOSE

        std::cout << title << ": AbsoluteErr=" << nrmDiff <<\
                " RelativeErr=" << errRel << "\n" << std::endl;
#ifndef VVERBOSE
      }
#endif VVERBOSE
#endif


      return errRel;
}





};





CPPUNIT_TEST_SUITE_REGISTRATION( CusplaTestCase );

int main(int argc, char** argv)
{

    CppUnit::TextUi::TestRunner runner;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    runner.addTest( registry.makeTest() );
    runner.run();
    return 0;

}



