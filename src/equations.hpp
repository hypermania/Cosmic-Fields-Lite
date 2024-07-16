/*! 
  \file equations.hpp
  \brief Header for field equations that runs on the CPU.

  This is the header for field equations that are supposed to run on CPU.
  Currently only the Klein Gordon equation \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \varphi = 0 \f$
  and the FRW Klein Gordon equation \f$ \ddot{\varphi} + 3 H \dot{\varphi} - \nabla^2 \varphi / a^2 + m^2 \varphi = 0 \f$ are implemented.
  
  Each equation struct implements both \b operator() and \b compute_energy_density(const Workspace &, const double), which computes and save the time derivative and computes the energy density of the equation.

*/

#ifndef EQUATIONS_H
#define EQUATIONS_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <cmath>

#include "Eigen/Dense"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

#include "odeint_eigen/eigen_operations.hpp"

#include "fdm3d.hpp"
#include "io.hpp"
#include "physics.hpp"
#include "workspace.hpp"


// #ifndef DISABLE_CUDA
// #include "cuda_wrapper.cuh"
// #include "fdm3d_cuda.cuh"
// #endif


template<typename Equation>
concept LatticeEquationConcept = requires (Equation eqn)
  {
   //typename Equation::State;
   eqn.workspace;
   eqn.compute_energy_density(eqn.workspace, 0.0);
  };


/*! 
  \struct KleinGordonEquation
  \brief The Klein Gordon equation.

  Defines the Klein Gordon equation \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \varphi = 0 \f$.
  \f{eqnarray*}{
  \Huge
  \ddot{\varphi} - \nabla^2 \varphi + m^2 \varphi = 0
  \f}

  \param[out] test The memory area to copy to.
*/
struct KleinGordonEquation {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  KleinGordonEquation(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


struct KleinGordonEquationInFRW {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  KleinGordonEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


struct ComovingCurvatureEquationInFRW {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  ComovingCurvatureEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(Workspace &workspace, const double t);
};




#endif
