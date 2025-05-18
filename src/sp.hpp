/*
  Tools related to Schrodinger-Poisson fields.
*/
#ifndef SP_HPP
#define SP_HPP

#include "Eigen/Dense"
#include "random_field.hpp"
#include "dispatcher.hpp"
#include "workspace.hpp"
#include "fdm3d.hpp"

// Our convention for the state of a Proca field is a vector [Ax, Ay, Az, dt_Ax, dt_Ay, dt_Az].
// Namely, 3 copies of a Klein Gordon field.

/*! 
  \brief Generate 3 concatenated Gaussian random fields. Namely a Proca field.
  \param N Number of lattice points.
  \param L Box size.
  \param P The spectrum \f$ P \f$.
  \return The generated GRF, as values on the lattice (of size \f$ 3 N^3 \f$).

  Generate 3 Gaussian random fields \f$ [A_x, A_y, A_z] \f$, such that the spectrum of \f$ A_i \f$ is \f$ P \f$.
*/
Eigen::VectorXd generate_gaussian_random_sp_field(const long long int N, const double L, const Spectrum &P);


/*! \brief Initialize a Schrodinger-Poisson field and its derivative from a white noise power spectrum with cutoff k_ast. */
inline auto unperturbed_sp_grf =
  [](const auto param, auto &workspace) {
    const long long int lattice_size = param.N * param.N * param.N;
    
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    
    // The code is CPU only
    auto &state = workspace.state;
    state.resize(6 * lattice_size);
    state.segment(0, 3 * lattice_size) = generate_gaussian_random_proca_field(param.N, param.L, P_f);
    state.segment(3 * lattice_size, 3 * lattice_size) = generate_gaussian_random_proca_field(param.N, param.L, P_dtf);
    
  };


/*! 
  \brief The SchrodingerPoisson equation, \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \vec{A} = 0 \f$.
*/
struct SchrodingerPoissonEquation {
  typedef Eigen::VectorXcd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  SchrodingerPoissonEquation(Workspace &workspace_) : workspace(workspace_) {}

  /*!
    \brief The function called by odeint library.
    \param[in] x The current state of the system.
    \param[out] dxdt The time derivative, dxdt of the system.
    \param t The current time parameter.
  */
  void operator()(const State &, State &, const double);


  /*!
    \brief Compute the time component of the Proca field from the workspace.
    \param[in] workspace The workspace for evaluating the energy density.
    \param t The current time parameter.
    \return A vector of size \f$ N^3 \f$.
  */
  static Vector compute_At(Workspace &workspace, const double t);
    
  /*!
    \brief Compute the energy density profile from the workspace.
    \param[in] workspace The workspace for evaluating the energy density.
    \param t The current time parameter.
    \return A vector of size \f$ N^3 \f$, giving the energy density profile \f$ \rho = \frac12 (\dot{\varphi}^2 + (\nabla\varphi)^2 + m^2 \varphi^2 \f$ on the lattice.
  */
  static Vector compute_energy_density(Workspace &workspace, const double t);


  /*!
    \brief Compute the momentum density profile from the workspace.
    \param[in] workspace The workspace for evaluation.
    \param t The current time parameter.
    \return A vector of size \f$ 3 N^3 \f$, giving the momentum density profile \f$ {\bf q} = - \dot{\varphi} \nabla\varphi \f$ on the lattice in x, y, z directions.
  */
  static Vector compute_momentum_density(Workspace &workspace, const double t);
};

#endif
