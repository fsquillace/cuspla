/*
 * utils.h
 *
 *  Created on: Feb 25, 2012
 *      Author: Filippo Squillace
 */


#include <thrust/functional.h>

namespace cuspla{

struct in_diagonal : public thrust::unary_function<int,bool>
{
    int N, M;

    __host__ __device__
    in_diagonal(int _N, int _M) : N(_N), M(_M) {}

    __host__ __device__
    bool operator()(int i)
    {
        if (i %(N+1) ==0)
            return true;
        else
            return false;
    }
};


template <typename ValueType>
struct mul_const : public thrust::unary_function<ValueType,ValueType>
{
	ValueType alpha;

	mul_const(ValueType _alpha)
		: alpha(_alpha) {}

	__host__ __device__
		void operator()(ValueType & x)
		{
			x = alpha * x;
		}
};


template <typename ValueType>
struct plus_const : public thrust::unary_function<ValueType,ValueType>
{
    ValueType cons;

    __host__ __device__
    plus_const(ValueType _const) : cons(_const) {}

    __host__ __device__
    ValueType operator()(ValueType e)
    {
        return e+cons;
    }
};

template <typename ValueType>
struct assigns : public thrust::unary_function<ValueType,ValueType>
{
    ValueType cons;

    __host__ __device__
    assigns(ValueType _const) : cons(_const) {}

    __host__ __device__
    ValueType operator()(ValueType e)
    {
        return cons;
    }
};

template <typename ValueType>
struct copy : public thrust::unary_function<ValueType,ValueType>
{

    __host__ __device__
    copy() {}

    __host__ __device__
    ValueType operator()(ValueType e)
    {
        return e;
    }
};

struct in_upper_triang : public thrust::unary_function<int,bool>
{
    int N, M;

    __host__ __device__
    in_upper_triang(int _N, int _M) : N(_N), M(_M) {}

    __host__ __device__
    bool operator()(size_t num)
    {
        size_t i= num%N;
        size_t j=num/N;

        if (j>=i)
            return true;
        else
            return false;
    }
};


}
