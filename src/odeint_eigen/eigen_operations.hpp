/*
  Modifications by Siyang Ling.
  Adapted so that Eigen can work with runge_kutta_dopri5 and runge_kutta_fehlberg78.
  The generic Runge Kutta algorithms implemented by odeint uses Boost.fusion, which introduces extra memory allocation/free. The "scale_sum*" operations are reimplemented here using native Eigen to avoid this overhead.
*/

#ifndef EIGEN_OPERATIONS_HPP
#define EIGEN_OPERATIONS_HPP

#include <Eigen/Dense>

namespace boost {
  namespace numeric {
    namespace odeint {
      struct eigen_operations {

	template<class Fac = double>
	struct scale_sum1
	{
	  const Fac m_alpha1;

	  scale_sum1(const Fac alpha1)
	    : m_alpha1(alpha1) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1) const
	  {
	    v0 = m_alpha1 * v1;
	  }
	};

	
	template<class Fac = double, class Time1 = Fac>
	struct scale_sum2
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;

	  scale_sum2(const Fac alpha1, const Fac alpha2)
	    : m_alpha1(alpha1) , m_alpha2(alpha2) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2;
	  }
	};


	// The extra Time? template parameters are required for dopri5 to work.
	template<class Fac = double, class Time1 = Fac, class Time2 = Fac>
	struct scale_sum3
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;

	  scale_sum3(const Fac alpha1, const Fac alpha2, const Fac alpha3)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3;
	  }
	};

	
	//template<class Fac = double, class Time = double>
	template<class Fac = double, class Time1 = Fac, class Time2 = Fac, class Time3 = Fac>
	struct scale_sum4
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;

	  scale_sum4(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4;
	  }
	};


	//template<class Fac = double, class Time = double>
	template<class Fac = double, class Time1 = Fac, class Time2 = Fac, class Time3 = Fac, class Time4 = Fac>
	struct scale_sum5
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;

	  scale_sum5(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5;
	  }
	};

	
	//template<class Fac = double, class Time = double>
	template<class Fac = double, class Time1 = Fac, class Time2 = Fac, class Time3 = Fac, class Time4 = Fac, class Time5 = Fac>
	struct scale_sum6
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;

	  scale_sum6(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6;
	  }
	};


	//template<class Fac = double, class Time = double>
	template<class Fac = double, class Time1 = Fac, class Time2 = Fac, class Time3 = Fac, class Time4 = Fac, class Time5 = Fac, class Time6 = Fac>
	struct scale_sum7
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;

	  scale_sum7(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum8
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;

	  scale_sum8(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum9
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;
	  const Fac m_alpha9;

	  scale_sum9(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8, const Fac alpha9)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8), m_alpha9(alpha9) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8, const Eigen::VectorXd &v9) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8 + m_alpha9 * v9;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum10
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;
	  const Fac m_alpha9;
	  const Fac m_alpha10;

	  scale_sum10(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8, const Fac alpha9, const Fac alpha10)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8), m_alpha9(alpha9), m_alpha10(alpha10) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8, const Eigen::VectorXd &v9, const Eigen::VectorXd &v10) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8 + m_alpha9 * v9 + m_alpha10 * v10;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum11
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;
	  const Fac m_alpha9;
	  const Fac m_alpha10;
	  const Fac m_alpha11;

	  scale_sum11(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8, const Fac alpha9, const Fac alpha10, const Fac alpha11)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8), m_alpha9(alpha9), m_alpha10(alpha10), m_alpha11(alpha11) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8, const Eigen::VectorXd &v9, const Eigen::VectorXd &v10, const Eigen::VectorXd &v11) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8 + m_alpha9 * v9 + m_alpha10 * v10 + m_alpha11 * v11;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum12
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;
	  const Fac m_alpha9;
	  const Fac m_alpha10;
	  const Fac m_alpha11;
	  const Fac m_alpha12;

	  scale_sum12(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8, const Fac alpha9, const Fac alpha10, const Fac alpha11, const Fac alpha12)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8), m_alpha9(alpha9), m_alpha10(alpha10), m_alpha11(alpha11), m_alpha12(alpha12) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8, const Eigen::VectorXd &v9, const Eigen::VectorXd &v10, const Eigen::VectorXd &v11, const Eigen::VectorXd &v12) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8 + m_alpha9 * v9 + m_alpha10 * v10 + m_alpha11 * v11 + m_alpha12 * v12;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum13
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;
	  const Fac m_alpha9;
	  const Fac m_alpha10;
	  const Fac m_alpha11;
	  const Fac m_alpha12;
	  const Fac m_alpha13;

	  scale_sum13(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8, const Fac alpha9, const Fac alpha10, const Fac alpha11, const Fac alpha12, const Fac alpha13)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8), m_alpha9(alpha9), m_alpha10(alpha10), m_alpha11(alpha11), m_alpha12(alpha12), m_alpha13(alpha13) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8, const Eigen::VectorXd &v9, const Eigen::VectorXd &v10, const Eigen::VectorXd &v11, const Eigen::VectorXd &v12, const Eigen::VectorXd &v13) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8 + m_alpha9 * v9 + m_alpha10 * v10 + m_alpha11 * v11 + m_alpha12 * v12 + m_alpha13 * v13;
	  }
	};


	template<class Fac = double, class Time = double>
	struct scale_sum14
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;
	  const Fac m_alpha3;
	  const Fac m_alpha4;
	  const Fac m_alpha5;
	  const Fac m_alpha6;
	  const Fac m_alpha7;
	  const Fac m_alpha8;
	  const Fac m_alpha9;
	  const Fac m_alpha10;
	  const Fac m_alpha11;
	  const Fac m_alpha12;
	  const Fac m_alpha13;
	  const Fac m_alpha14;

	  scale_sum14(const Fac alpha1, const Fac alpha2, const Fac alpha3, const Fac alpha4, const Fac alpha5, const Fac alpha6, const Fac alpha7, const Fac alpha8, const Fac alpha9, const Fac alpha10, const Fac alpha11, const Fac alpha12, const Fac alpha13, const Fac alpha14)
	    : m_alpha1(alpha1) , m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5), m_alpha6(alpha6), m_alpha7(alpha7), m_alpha8(alpha8), m_alpha9(alpha9), m_alpha10(alpha10), m_alpha11(alpha11), m_alpha12(alpha12), m_alpha13(alpha13), m_alpha14(alpha14) {}

	  void operator()(Eigen::VectorXd &v0, const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4, const Eigen::VectorXd &v5, const Eigen::VectorXd &v6, const Eigen::VectorXd &v7, const Eigen::VectorXd &v8, const Eigen::VectorXd &v9, const Eigen::VectorXd &v10, const Eigen::VectorXd &v11, const Eigen::VectorXd &v12, const Eigen::VectorXd &v13, const Eigen::VectorXd &v14) const
	  {
	    v0 = m_alpha1 * v1 + m_alpha2 * v2 + m_alpha3 * v3 + m_alpha4 * v4 + m_alpha5 * v5 + m_alpha6 * v6 + m_alpha7 * v7 + m_alpha8 * v8 + m_alpha9 * v9 + m_alpha10 * v10 + m_alpha11 * v11 + m_alpha12 * v12 + m_alpha13 * v13 + m_alpha14 * v14;
	  }
	};


	template<class Fac = double, class Time1 = Fac>
	struct scale_sum_swap2
	{
	  const Fac m_alpha1;
	  const Fac m_alpha2;

	  scale_sum_swap2(const Fac alpha1, const Fac alpha2)
	    : m_alpha1(alpha1) , m_alpha2(alpha2) {}

	  void operator()(Eigen::VectorXd &t1, Eigen::VectorXd &t2, const Eigen::VectorXd &t3) const
	  {
	    t1.swap(t2);
	    t1 = m_alpha1 * t1 + m_alpha2 * t3;
	  }
	};

	
	template<class Fac = double>
	struct rel_error
	{
	  const Fac m_eps_abs, m_eps_rel, m_a_x, m_a_dxdt;

	  rel_error(Fac eps_abs, Fac eps_rel, Fac a_x, Fac a_dxdt)
	    : m_eps_abs(eps_abs), m_eps_rel(eps_rel), m_a_x(a_x), m_a_dxdt(a_dxdt) {}

	  void operator()(Eigen::VectorXd &x_err, const Eigen::VectorXd &x_old, const Eigen::VectorXd &dxdt_old) const
	  {
	    x_err.array() = x_err.array().abs() / (m_eps_abs + m_eps_rel * (m_a_x * x_old.array().abs() + m_a_dxdt * dxdt_old.array().abs()));
	  }
	};
	
      };
      
      template<>
      // specialization for eigen vectors
      struct operations_dispatcher<typename Eigen::VectorXd>
      {
	typedef eigen_operations operations_type;
      };

    } // namespace odeint
  } // namespace numeric
} // namespace boost


namespace Eigen {

  inline const Eigen::VectorXd
  operator/(const int &i, const Eigen::VectorXd &m) {
    return ((double)i) * m.cwiseInverse();
  }

  /*
  template<typename D>
  inline const 
  auto
  pow(const Eigen::MatrixBase<D> &m, const double exp) {
    return m.array().pow(exp).matrix();
  }
  */
  
  /*
  template< typename D >
  inline const 
  typename Eigen::CwiseUnaryOp<
    typename Eigen::internal::scalar_abs_op<
      typename Eigen::internal::traits< D >::Scalar > ,
    const D >
  pow(const Eigen::MatrixBase<D> &m, const double exp) {
    return m.array().pow(exp).matrix();
  }
  */
  
  /*
  template<typename D1,typename D2>
  inline const
  typename Eigen::CwiseBinaryOp<
    typename Eigen::internal::scalar_quotient_op<
      typename Eigen::internal::traits<D1>::Scalar>,
    const D1, const D2>
  operator=(Eigen::MatrixBase<D1> &x1, const Eigen::MatrixBase<D2> &x2) {
    return x1.cwiseQuotient(x2);
  }
  */
    


} // end Eigen namespace

#endif
