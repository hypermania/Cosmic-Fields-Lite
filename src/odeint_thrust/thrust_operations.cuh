/*
  Adapted from:  
  boost/numeric/odeint/external/thrust/thrust_operations.hpp
*/


#ifndef THRUST_OPERATIONS_CUH
#define THRUST_OPERATIONS_CUH

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// #include <thrust/fill.h>
// #include <thrust/transform.h>

#include "../cuda_wrapper.cuh"

namespace boost {
namespace numeric {
namespace odeint {

struct thrust_operations
{
  typedef thrust::device_vector<double> State;

  
  template<class Fac1 = double>
  struct scale_sum1
  {
    const Fac1 m_alpha1;

    scale_sum1(const Fac1 alpha1);
    
    void operator()(State &v0, const State &v1) const;
  };

  
  template<class Fac1 = double, class Fac2 = Fac1>
  struct scale_sum2
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;

    scale_sum2(const Fac1 alpha1, const Fac2 alpha2);
    
    void operator()(State &v0, const State &v1, const State &v2) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2>
  struct scale_sum3
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;

    scale_sum3(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3);
    
    void operator()(State &v0, const State &v1, const State &v2, const State &v3) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3>
  struct scale_sum4
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;

    scale_sum4(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4>
  struct scale_sum5
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;

    scale_sum5(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
	       const Fac5 alpha5);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5>
  struct scale_sum6
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;

    scale_sum6(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
	       const Fac5 alpha5, const Fac6 alpha6);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6) const;
  };

  
  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6>
  struct scale_sum7
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;

    scale_sum7(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
	       const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7);
    
    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7) const;
  };

  
  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7>
  struct scale_sum8
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;

    scale_sum8(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
	       const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7,
	   class Fac9 = Fac8>
  struct scale_sum9
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;
    const Fac9 m_alpha9;

    scale_sum9(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
	       const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8,
	       const Fac9 alpha9);
    
    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8, const State &v9) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7,
	   class Fac9 = Fac8, class Fac10 = Fac9>
  struct scale_sum10
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;
    const Fac9 m_alpha9;
    const Fac10 m_alpha10;

    scale_sum10(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
		const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8,
		const Fac9 alpha9, const Fac10 alpha10);
    
    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8, const State &v9, const State &v10) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7,
	   class Fac9 = Fac8, class Fac10 = Fac9, class Fac11 = Fac10>
  struct scale_sum11
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;
    const Fac9 m_alpha9;
    const Fac10 m_alpha10;
    const Fac11 m_alpha11;

    scale_sum11(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
		const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8,
		const Fac9 alpha9, const Fac10 alpha10, const Fac11 alpha11);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8, const State &v9, const State &v10, const State &v11) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7,
	   class Fac9 = Fac8, class Fac10 = Fac9, class Fac11 = Fac10, class Fac12 = Fac11>
  struct scale_sum12
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;
    const Fac9 m_alpha9;
    const Fac10 m_alpha10;
    const Fac11 m_alpha11;
    const Fac12 m_alpha12;

    scale_sum12(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
		const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8,
		const Fac9 alpha9, const Fac10 alpha10, const Fac11 alpha11, const Fac12 alpha12);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8, const State &v9, const State &v10, const State &v11,
		    const State &v12) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7,
	   class Fac9 = Fac8, class Fac10 = Fac9, class Fac11 = Fac10, class Fac12 = Fac11,
	   class Fac13 = Fac12>
  struct scale_sum13
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;
    const Fac9 m_alpha9;
    const Fac10 m_alpha10;
    const Fac11 m_alpha11;
    const Fac12 m_alpha12;
    const Fac13 m_alpha13;

    scale_sum13(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
		const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8,
		const Fac9 alpha9, const Fac10 alpha10, const Fac11 alpha11, const Fac12 alpha12,
		const Fac13 alpha13);

    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8, const State &v9, const State &v10, const State &v11,
		    const State &v12, const State &v13) const;
  };


  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
	   class Fac5 = Fac4, class Fac6 = Fac5, class Fac7 = Fac6, class Fac8 = Fac7,
	   class Fac9 = Fac8, class Fac10 = Fac9, class Fac11 = Fac10, class Fac12 = Fac11,
	   class Fac13 = Fac12, class Fac14 = Fac13>
  struct scale_sum14
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;
    const Fac6 m_alpha6;
    const Fac7 m_alpha7;
    const Fac8 m_alpha8;
    const Fac9 m_alpha9;
    const Fac10 m_alpha10;
    const Fac11 m_alpha11;
    const Fac12 m_alpha12;
    const Fac13 m_alpha13;
    const Fac14 m_alpha14;

    scale_sum14(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3, const Fac4 alpha4,
		const Fac5 alpha5, const Fac6 alpha6, const Fac7 alpha7, const Fac8 alpha8,
		const Fac9 alpha9, const Fac10 alpha10, const Fac11 alpha11, const Fac12 alpha12,
		const Fac13 alpha13, const Fac14 alpha14);
    
    void operator()(State &v0, const State &v1, const State &v2, const State &v3,
		    const State &v4, const State &v5, const State &v6, const State &v7,
		    const State &v8, const State &v9, const State &v10, const State &v11,
		    const State &v12, const State &v13, const State &v14) const;
  };

  
  template<class Fac1 = double, class Fac2 = Fac1>
  struct scale_sum_swap2
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;

    scale_sum_swap2(const Fac1 alpha1, const Fac2 alpha2);
    
    void operator()(State &v0, State &v1, const State &v2) const;
  };

  
  template<class Fac = double>
  struct rel_error
  {
    const Fac m_eps_abs, m_eps_rel, m_a_x, m_a_dxdt;

    rel_error(Fac eps_abs, Fac eps_rel, Fac a_x, Fac a_dxdt);
    
    void operator()(State &x_err, const State &x_old, const State &dxdt_old) const;
  };
	
};

} // odeint
} // numeric
} // boost


#endif
