/*
  Tools related to Proca fields.
*/
#ifndef PROCA_HPP
#define PROCA_HPP

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
Eigen::VectorXd generate_gaussian_random_proca_field(const long long int N, const double L, const Spectrum &P);

struct ProcaTransverseProjector {
  ProcaTransverseProjector(long long int N_) :
    N(N_), lattice_size(N*N*N), fourier_size(2*N*N*(N/2+1))
  { init(); }

  void init(void);

  /*! 
    \brief Generate 3 concatenated Gaussian random fields. Namely a Proca field.
    \param fields A Proca field. Namely a vector of length \f$ 3 N^3 \f$.
    \param fft_wrapper
    Modify the Proca field, so that only the transverse components are retained.
  */
  void proca_project_to_transverse(Eigen::VectorXd &fields, fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

  long long int N;
  long long int lattice_size;
  long long int fourier_size;
  
  Eigen::VectorXd M_xx_k;
  Eigen::VectorXd M_xy_k;
  Eigen::VectorXd M_xz_k;
  // Eigen::VectorXd M_yx_k;
  Eigen::VectorXd M_yy_k;
  Eigen::VectorXd M_yz_k;
  // Eigen::VectorXd M_zx_k;
  // Eigen::VectorXd M_zy_k;
  Eigen::VectorXd M_zz_k;
};


/*! \brief Initialize a Proca field and its derivative from a white noise power spectrum with cutoff k_ast. */
inline auto unperturbed_proca_grf =
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
  \brief The Proca equation, \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \vec{A} = 0 \f$.
*/
struct ProcaEquation {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  ProcaEquation(Workspace &workspace_) : workspace(workspace_) {}

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

void check_proca_q(void);

#endif
