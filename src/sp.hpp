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
  \brief Generate 3 concatenated Gaussian random SP fields.
  \param N Number of lattice points.
  \param L Box size.
  \param P The spectrum \f$ P \f$.
  \return The generated GRF, as complex values on the lattice (of size \f$ 3 N^3 \f$).

  Generate 3 Gaussian random fields \f$ [\psi_1, \psi_2, \psi_3] \f$, such that the spectrum of \f$ \psi_i \f$ is \f$ P \f$.
*/
Eigen::ArrayXcd generate_gaussian_random_sp_field(const long long int N, const double L, const Spectrum &P);


/*! 
  \brief Generate 3 concatenated Gaussian random SP fields.
  \param N Number of lattice points.
  \param L Box size.
  \param f The inhomogeneity function \f$ f \f$, given in terms of values on the lattice (of size \f$ N^3 \f$).
  \param P The spectrum \f$ P \f$.
  \return The generated GRF, as complex values on the lattice (of size \f$ 3 N^3 \f$).

  Generate 3 Gaussian random fields \f$ [\psi_1, \psi_2, \psi_3] \f$, such that the spectrum of \f$ \psi_i \f$ is \f$ P \f$, and the variance varies like \f$ e^{2 f} \f$.
*/
Eigen::ArrayXcd generate_inhomogeneous_gaussian_random_sp_field(const long long int N, const double L, const Eigen::VectorXd &f, const Spectrum &P);


/*! \brief Initialize a Schrodinger-Poisson field and its derivative from a white noise power spectrum with cutoff k_ast. */
inline auto unperturbed_sp_grf =
  [](const auto param, auto &workspace) {
    const long long int lattice_size = param.N * param.N * param.N;
    
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);

    // The code is CPU only
    auto &state = workspace.state;
    state = generate_gaussian_random_sp_field(param.N, param.L, P_f);

    workspace.Psi.resize(lattice_size);
    workspace.Psi.array() = 0;
  };

/*! \brief Initialize a Schrodinger-Poisson field and its derivative from a white noise power spectrum with cutoff k_ast. */
inline auto matter_dominated_sp_grf =
  [](const auto param, auto &workspace) {
    const auto N = param.N;
    const auto L = param.L;
    const auto a1 = param.a1;
    const long long int lattice_size = N*N*N;
    
    auto &fft_wrapper = workspace.fft_wrapper; // fftwWrapper(N);
    
    double A_s = 1e-9;
    double ai = 1;

    Spectrum P_R = scale_invariant_spectrum_3d(N, L, A_s);
    Eigen::VectorXd R = generate_gaussian_random_field(N, L, P_R);
    workspace.R_fft = fft_wrapper.execute_d2z(R);
    
    // Convention for potentials: \mathcal{R}_k = (3 / 2) \Psi_k for superhorizon.
    // This works both sub and superhorizon, but only works in radiation era.
    // For potential during matter domination I need another potential.
    double eta_i = workspace.cosmology.eta(param.t_start);
    double k_eq = 1; // We use units in which a_eq = 1, H_eq = 1
    auto Phi_kernel = [&](double k){
      return k == 0.0 ? 0.0 : (k < k_eq ? 0.6 : 0.0);
    };
    auto dot_Phi_kernel = [&](double k){
      return k == 0.0;
    };
    auto delta_kernel = [&](double k){
      return k == 0.0 ? 0.0 : (k < k_eq ? (-pow(ai, -2) * a1) : (-pow(k,-2) * pow(ai, -2) * a1));
    };
    auto dot_delta_kernel = [&](double k){
      return k == 0.0 ? 0.0 : (k < k_eq ? (-pow(ai, -2) * pow(a1, -0.5)) : (-pow(k,-2) * pow(ai, -2) * pow(a1, -0.5)));
    };

    Eigen::VectorXd half_f(lattice_size);
    Eigen::VectorXd tau(lattice_size);
    {
      Eigen::VectorXd Phi = compute_field_with_scaled_fourier_modes(N, L, R, Phi_kernel, fft_wrapper);
      Eigen::VectorXd delta = compute_field_with_scaled_fourier_modes(N, L, R, delta_kernel, fft_wrapper);
      half_f = 0.5 * (delta + Phi);
    }
    std::cout << "half_f : " << half_f.head(16).transpose() << std::endl;
    {
      // Eigen::VectorXd tau_RHS = pow(a1, 2) * (dot_delta - 3 * dot_Phi);
      Eigen::VectorXd dot_delta = compute_field_with_scaled_fourier_modes(N, L, R, dot_delta_kernel, fft_wrapper);
      Eigen::VectorXd tau_RHS = pow(a1, 2) * dot_delta;
      tau = compute_inverse_laplacian(N, L, tau_RHS, fft_wrapper);
    }
    workspace.Psi = tau;
    std::cout << "half_f.norm() = " << half_f.norm() << '\n';
    std::cout << "tau.norm() = " << tau.norm() << '\n';
    
    // The code is CPU only
    auto &state = workspace.state;
    Spectrum P_psi = power_law_with_cutoff_given_amplitude_3d(N, L, param.varphi_std_dev, param.k_ast, 0);
    state = generate_inhomogeneous_gaussian_random_sp_field(N, L, half_f, P_psi);

    std::cout << "state : " << state.abs2().head(16).transpose() << std::endl;

  };


/*! 
  \brief The SchrodingerPoisson equation, \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \vec{A} = 0 \f$.
*/
struct SchrodingerPoissonEquation {
  typedef Eigen::ArrayXcd State;
  typedef Eigen::VectorXd Vector;
  typedef WorkspaceGeneric<Vector, State> Workspace;
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
