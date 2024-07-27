/*!
  \file physics.hpp
  \brief Collection of repeatedly used physics formulas. (e.g. FRW cosmology related formulas)
 
*/
#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#include <cmath>
//#include "param.hpp"

struct StaticEOSCosmology {
  double a1;
  double H1;
  double t1;
  double p;

  StaticEOSCosmology(const double a1_, const double H1_, const double t1_, const double p_)
    : a1(a1_), H1(H1_), t1(t1_), p(p_) {}

  // The default constructor from a param assumes radiation domination
  template<typename T>
  StaticEOSCosmology(const T &param)
    : a1(param.a1), H1(param.H1), t1(param.t1), p(1.0) {}
  
  StaticEOSCosmology(void)
    : a1(1.0), H1(0), t1(0), p(1.0) {}

  double a(const double t) const {
    return a1 * pow(1 + (1 + 1 / p) * H1 * (t - t1), p / (1 + p));
  }
  double H(const double t) const {
    return H1 * pow(1 + (1 + 1 / p) * H1 * (t - t1), -1);
  }
  // We use convention eta1 = p / (a1 * H1).
  double eta(const double t) const {
    //return eta1 + (p / (a1 * H1)) * (-1 + pow(1 + (1 + 1 / p) * H1 * (t - t1), 1 / (1 + p)));
    return (p / (a1 * H1)) * pow(1 + (1 + 1 / p) * H1 * (t - t1), 1 / (1 + p));
  }
};


#endif
