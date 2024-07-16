/*!
  \file wkb.hpp
  \brief Implementation of the WKB solution.
 
  Used to extend an existing field profile to a later time.
*/
#ifndef WKB_H
#define WKB_H

#include <cmath>

#include "fftw_wrapper.hpp"
#include "physics.hpp"
#include "workspace.hpp"

//Eigen::VectorXd wkb_solution(const Eigen::VectorXd &state, StaticEOSCosmology &cosmology, double ti, double t);

struct WKBSolutionForKleinGordonEquationInFRW {

  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  
  Workspace &workspace;
  double t_i;
  Vector phi_ffts;
  
  WKBSolutionForKleinGordonEquationInFRW(Workspace &workspace_, const double t_i_);
  
  Vector evaluate_at(const double t);

};

#endif
