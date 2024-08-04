/*!
  \file physics.hpp
  \author Siyang Ling
  \brief Collection of repeatedly used physics formulas. (e.g. FRW cosmology related formulas)
*/
#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#include <cmath>
//#include "param.hpp"

/*!
  \brief A convenience class used to calculate FRW related quantities for constant EOS spacetimes.
*/
struct StaticEOSCosmology {
  double a1; /*!< The scale factor at \f$ t_1 \f$. */
  double H1; /*!< The Hubble parameter at \f$ t_1 \f$. */
  double t1; /*!< A pivot coordinate time \f$ t_1 \f$. */
  double p; /*!< Power-law between scale factor and conformal time, \f$ a \propto \eta^p\f$. In terms of EOS \f$ w \f$, we have \f$ p = \frac{2}{1+3w}\f$. */

  StaticEOSCosmology(const double a1_, const double H1_, const double t1_, const double p_)
    : a1(a1_), H1(H1_), t1(t1_), p(p_) {}

  /*! \brief This constructor assumes radiation domination. */
  template<typename T>
  StaticEOSCosmology(const T &param)
    : a1(param.a1), H1(param.H1), t1(param.t1), p(1.0) {}
  
  /*! \brief The default constructor gives Minkowski spacetime. */
  StaticEOSCosmology(void)
    : a1(1.0), H1(0), t1(0), p(1.0) {}

  /*! \brief Returns the scale factor at coordinate time `t`. */
  double a(const double t) const {
    return a1 * pow(1 + (1 + 1 / p) * H1 * (t - t1), p / (1 + p));
  }

  /*! \brief Returns the Hubble parameter at coordinate time `t`. */
  double H(const double t) const {
    return H1 * pow(1 + (1 + 1 / p) * H1 * (t - t1), -1);
  }

  /*! 
    \brief Returns the conformal time \f$ \eta \f$ at coordinate time \f$ t \f$.

    Converts coordinate time \f$ t \f$ to conformal time \f$ \eta \f$.
    The conversion assumes the convention \f$ a = a_1 (\eta / \eta_1)^p \f$, where \f$ \eta_1 = p / (a_1 H_1) \f$.
    In this convention, \f$ \eta_1 \f$ and \f$ t_1 \f$ are at the same physical time.
    The conversion formula is 
    \f[ \eta = \frac{p}{a_1 H_1} \left(1 + (1+1/p) H_1 (t-t_1)\right)^{1/(1+p)}\f].
  */
  double eta(const double t) const {
    //return eta1 + (p / (a1 * H1)) * (-1 + pow(1 + (1 + 1 / p) * H1 * (t - t1), 1 / (1 + p)));
    return (p / (a1 * H1)) * pow(1 + (1 + 1 / p) * H1 * (t - t1), 1 / (1 + p));
  }
};


#endif
