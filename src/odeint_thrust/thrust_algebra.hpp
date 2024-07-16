/*
 [auto_generated]
 boost/numeric/odeint/external/thrust/thrust_algebra.hpp

 [begin_description]
 An algebra for thrusts device_vectors.
 [end_description]

 Copyright 2010-2013 Mario Mulansky
 Copyright 2010-2011 Karsten Ahnert
 Copyright 2013 Kyle Lutz

 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef BOOST_NUMERIC_ODEINT_EXTERNAL_THRUST_THRUST_ALGEBRA_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_EXTERNAL_THRUST_THRUST_ALGEBRA_HPP_INCLUDED


#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <boost/range.hpp>

// Overload for vector_space_algebra::norm_inf
namespace boost {
  namespace numeric {
    namespace odeint {

      namespace detail {

	// to use in thrust::reduce
	template< class Value >
	struct maximum
	{
	  template< class Fac1 , class Fac2 >
	  __host__ __device__
	  Value operator()( const Fac1 t1 , const Fac2 t2 ) const
	  {
            return ( abs( t1 ) < abs( t2 ) ) ? t2 : t1 ;
	  }

	  typedef Value result_type;
	};
      }
    
      template< class S >
      struct vector_space_norm_inf<S>
      {
	//typedef typename S::value_type value_type;
	typedef typename S::value_type result_type;
	//static typename S::value_type norm_inf( const S &s )
	result_type operator()(const S &s) const
	{
	  typedef typename S::value_type value_type;
	  return thrust::reduce( boost::begin( s ) , boost::end( s ) ,
				 static_cast<value_type>(0) ,
				 detail::maximum<value_type>() );
	}
      };


      /*
	struct abs_maximum
	{
	  __host__ __device__
	  double operator()( const double t1 , const double t2 ) const
	  {
            return ( abs( t1 ) < abs( t2 ) ) ? abs(t2) : abs(t1) ;
	  }
	};

	template<typename T>
	struct absolute_value : public thrust::unary_function<T,T>
	{
	  __host__ __device__ T operator()(const T &x) const
	  {
	    return x < T(0) ? -x : x;
	  }
	};
	
      }

      template<>
      struct vector_space_norm_inf<thrust::device_vector<double, thrust::device_allocator<double>>>
      {
	double operator()(const thrust::device_vector<double> &m) const
	{
	  return thrust::transform_reduce(m.begin(), m.end(),
					  detail::absolute_value<double>(), 0, thrust::maximum<double>());
	}
      };
	*/
      
      
    }
  }
}



#endif // BOOST_NUMERIC_ODEINT_EXTERNAL_THRUST_THRUST_ALGEBRA_HPP_INCLUDED
