/*! 
  \file equations.hpp
  \author Siyang Ling
  \brief Header for field equations that runs on the CPU.

  This is the header for field equations that are supposed to run on CPU.
  Equations declared here will be used by the odeint library via `operator()`.
  See <https://www.boost.org/doc/libs/1_85_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/getting_started/short_example.html> for an example of odeint equation.
  Typically, `compute_energy_density` is also implemented for saving energy density spectrum.
*/
#ifndef EQUATIONS_HPP
#define EQUATIONS_HPP


#include "Eigen/Dense"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

#include "odeint_eigen/eigen_operations.hpp"

#include "workspace.hpp"

template<typename Equation>
concept LatticeEquationConcept = requires (Equation eqn)
  {
   //typename Equation::State;
   eqn.workspace;
   eqn.compute_energy_density(eqn.workspace, 0.0);
  };


/*! 
  \brief The Klein Gordon equation, \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \varphi = 0 \f$.
*/
struct KleinGordonEquation {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  KleinGordonEquation(Workspace &workspace_) : workspace(workspace_) {}

  /*!
    \brief The function called by odeint library.
    \param[in] x The current state of the system.
    \param[out] dxdt The time derivative, dxdt of the system.
    \param t The current time parameter.
  */
  void operator()(const State &, State &, const double);

  /*!
    \brief Compute the energy density profile from the workspace.
    \param[in] workspace The workspace for evaluating the energy density.
    \param t The current time parameter.
    \return A vector of size \f$ N^3 \f$, giving the energy density profile \f$ \rho = \frac12 (\dot{\varphi}^2 + (\nabla\varphi)^2 + m^2 \varphi^2 \f$ on the lattice.
  */
  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


/*! 
  \brief The Klein Gordon in FRW equation, \f$ \ddot{\varphi} + 3 H \dot{\varphi} - \nabla^2 \varphi / a^2 + m^2 \varphi = 0 \f$.
*/
struct KleinGordonEquationInFRW {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  KleinGordonEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  /*!
    \brief Compute the energy density profile from the workspace.
    \param[in] workspace The workspace for evaluating the energy density.
    \param t The current time parameter.
    \return A vector of size \f$ N^3 \f$, giving the energy density profile \f$ \rho = \frac12 (\dot{\varphi}^2 + (\nabla\varphi)^2 / a(t)^2 + m^2 \varphi^2) \f$ given on the \f$ N^3 \f$ on the lattice.
  */
  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


/*! 
  \brief Equation for free scalar field in FRW spacetime, including comoving metric perturbations (in radiation domination).

  Equation is given by
  \f{align*}{
  & \ddot{\varphi} + 3 H \dot{\varphi} - e^{4\Psi} \frac{\nabla^2}{a^2} \varphi + e^{2\Psi} m^2 \varphi - 4 \dot{\Psi} \dot{\varphi} = 0 \\
  & \Psi_{\vb{k}}(t) = 2 \mathcal{R}_{\vb{k}} \frac{\sin(k \eta / \sqrt{3}) - (k \eta / \sqrt{3}) \cos(k \eta / \sqrt{3})}{(k \eta / \sqrt{3})^3} \\
  & \dot{\Psi}_{\vb{k}}(t) = 2 \mathcal{R}_{\vb{k}} H(t) \frac{3 (k \eta / \sqrt{3}) \cos(k \eta / \sqrt{3}) + ((k \eta / \sqrt{3})^2 - 3) \sin(k \eta / \sqrt{3})}{(k \eta / \sqrt{3})^3} \\
  & \eta(t) = \frac{(2 H_i t)^{1/2}}{a_i H_i} \qq{is the conformal time}
  \f}
  where \f$ \mathcal{R}_{\vb{k}} \f$ is read from workspace variable `R_fft`.
  See equation (6.160) of Baumann's cosmology textbook.
  This implementation is not optimized.
  It was only used for verifying the GPU implementations CudaComovingCurvatureEquationInFRW and CudaApproximateComovingCurvatureEquationInFRW.
*/
struct ComovingCurvatureEquationInFRW {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  ComovingCurvatureEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  /*!
    \brief Compute the energy density profile from the workspace.
    \param[in] workspace The workspace for evaluating the energy density.
    \param t The current time parameter.
    \return A vector of size \f$ N^3 \f$, giving the energy density profile \f$ \rho = \frac12 (e^{-2\Psi} \dot{\varphi}^2 + e^{2\Psi} (\nabla\varphi)^2 / a(t)^2 + m^2 \varphi^2) \f$ on the lattice.
  */
  static Vector compute_energy_density(Workspace &workspace, const double t);
};




#endif
