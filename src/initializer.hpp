/*!
  \file initializer.hpp
  \author Siyang Ling
  \brief Snippets for initializing workpaces. (e.g. field initial conditions, etc)

  Contains a collection of lambda functions used to initialize the workspace for simulation.
  Each lambda function `initializer` is meant to be passed to a workspace constructor `Workspace(param, initializer)`.
  The use of lambda's makes it easy to switch between `param` types and `workspace` types.
  Moreover, in order for easy switching between CPU code and GPU code, the initializers compute everything on CPU first,
  and then decide whether the results should be copied to CPU or GPU memory.
*/
#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include "fdm3d.hpp"
#include "random_field.hpp"
#include "fftw_wrapper.hpp"
#include "special_function.hpp"
//#include "dispatcher.hpp"
//#include "field_booster.hpp"
//#include "param.hpp"
//#include "physics.hpp"

#ifndef DISABLE_CUDA
#include <thrust/device_vector.h>
#include "cuda_wrapper.cuh"
#define ALGORITHM_NAMESPACE thrust
#else
#define ALGORITHM_NAMESPACE std
#endif

/*! \brief Initialize a field and its derivative from a white noise power spectrum with cutoff k_ast. */
inline auto unperturbed_grf =
  [](const auto param, auto &workspace) {
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd varphi = generate_gaussian_random_field(param.N, param.L, P_f); // Initial ULDM field
    Eigen::VectorXd dt_varphi = generate_gaussian_random_field(param.N, param.L, P_dtf); // Initial ULDM field time derivative
    
    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    // thrust::copy handles both copies between Eigen::VectorXd and copies from Eigen::VectorXd to thrust::device_vector<double>
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
  };


/*! \brief Initialize a field and its derivative from a white noise power spectrum with cutoff k_ast, plus homogeneous background. */
inline auto unperturbed_grf_with_background =
  [](const auto param, auto &workspace) {
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd varphi = generate_gaussian_random_field(param.N, param.L, P_f);
    varphi.array() += param.varphi_mean;
    Eigen::VectorXd dt_varphi = generate_gaussian_random_field(param.N, param.L, P_dtf);
    
    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
  };


/*!
  \brief Setup a scalar field with inhomogeneous Gaussian random initial conditions.

  Initialize a field and its derivative from a white noise power spectrum with cutoff k_ast,
  but with a large scale perturbation specified by Psi.
  Psi is initialized from a scale-invariant power spectrum with cutoff k_Psi.
*/
inline auto perturbed_grf =
  [](const auto param, auto &workspace) {
    Spectrum P_Psi = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd Psi = generate_gaussian_random_field(param.N, param.L, P_Psi);
    Eigen::VectorXd varphi = generate_inhomogeneous_gaussian_random_field(param.N, param.L, Psi, P_f);
    Eigen::VectorXd dt_varphi = generate_inhomogeneous_gaussian_random_field(param.N, param.L, Psi, P_dtf);

    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    workspace.Psi.resize(Psi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
    ALGORITHM_NAMESPACE::copy(Psi.begin(), Psi.end(), workspace.Psi.begin());

    //std::cout << boost::typeindex::type_id_runtime(workspace.Psi).pretty_name() << '\n';
  };


/*! \brief Same as perturbed_grf, but does not store `Psi` in `workspace` to save memory. */
inline auto perturbed_grf_without_saving_Psi =
  [](const auto param, auto &workspace) {
    Spectrum P_Psi = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd Psi = generate_gaussian_random_field(param.N, param.L, P_Psi);
    Eigen::VectorXd varphi = generate_inhomogeneous_gaussian_random_field(param.N, param.L, Psi, P_f);
    Eigen::VectorXd dt_varphi = generate_inhomogeneous_gaussian_random_field(param.N, param.L, Psi, P_dtf);

    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
  };


/*! \brief Same as unperturbed_grf, but with an extra scale-invariant `Psi`. */
inline auto unperturbed_grf_with_Psi =
  [](const auto param, auto &workspace) {
    Spectrum P_Psi = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd Psi = generate_gaussian_random_field(param.N, param.L, P_Psi);
    Eigen::VectorXd varphi = generate_gaussian_random_field(param.N, param.L, P_f);
    Eigen::VectorXd dt_varphi = generate_gaussian_random_field(param.N, param.L, P_dtf);

    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    workspace.Psi.resize(Psi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
    ALGORITHM_NAMESPACE::copy(Psi.begin(), Psi.end(), workspace.Psi.begin());
  };


/*! \brief Initialize a homogeneous Gaussian random field and some scale invariant curvature perturbation. */
inline auto unperturbed_grf_and_fixed_curvature =
  [](const auto param, auto &workspace) {
    Spectrum P_Psi = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd Psi = generate_gaussian_random_field(param.N, param.L, P_Psi);
    Eigen::VectorXd varphi = generate_gaussian_random_field(param.N, param.L, P_f);
    Eigen::VectorXd dt_varphi = generate_gaussian_random_field(param.N, param.L, P_dtf);
    
    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    workspace.Psi.resize(Psi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
    ALGORITHM_NAMESPACE::copy(Psi.begin(), Psi.end(), workspace.Psi.begin());
  };

/*! 
  \brief Initialize an inhomogeneous Gaussian random field and the fft of some scale invariant comoving curvature perturbation. 
  
  This is the procedure used for section 4.2.2 of the paper.
*/
inline auto perturbed_grf_and_comoving_curvature_fft =
  [](const auto param, auto &workspace) {
    using namespace std::numbers;

    // Generate comoving curvature perturbation
    double eta_i = workspace.cosmology.eta(param.t_start);
    double A_s = (-576 * pow(pi, 6) * pow(eta_i, 6) * pow(param.Psi_std_dev, 2)) /
      (-81 * pow(param.L, 4) * (pow(param.L, 2) + 2 * pow(pi, 2) * pow(eta_i, 2)) +
       param.L *
       (81 * pow(param.L, 5) - 54 * pow(param.L, 3) * pow(pi, 2) * pow(eta_i, 2) +
	48 * param.L * pow(pi, 4) * pow(eta_i, 4)) *
       cos((4 * pi * eta_i) / (sqrt(3) * param.L)) +
       256 * pow(pi, 6) * pow(eta_i, 6) * Ci_pade_approximant_12_12((4 * pi * eta_i) / (sqrt(3) * param.L)) +
       4 * sqrt(3) * param.L * pi * eta_i *
       (27 * pow(param.L, 4) + 6 * pow(param.L, 2) * pow(pi, 2) * pow(eta_i, 2) -
	16 * pow(pi, 4) * pow(eta_i, 4)) *
       sin((4 * pi * eta_i) / (sqrt(3) * param.L)));
    Spectrum P_R = scale_invariant_spectrum_3d(param.N, param.L, A_s);
    
    // Manual cutoff for P_R at around horizon. The effect of imposing this cutoff is negligible.
    // Spectrum P_R_with_cutoff = [P_R](double k){ return k <= 0.5 ? P_R(k) : 0.0; };
    
    Eigen::VectorXd R = generate_gaussian_random_field(param.N, param.L, P_R);
    // std::cout << "A_s = " << A_s << '\n';

    // Calculate initial gravitational potential Psi.
    // Convention for potentials: \mathcal{R}_k = (3 / 2) \Psi_k for superhorizon.
    auto kernel = [eta_i](double k){
		    return k == 0.0 ? 0.0 : (6 * sqrt(3) * (-((k * eta_i * cos((k * eta_i) / sqrt(3))) / sqrt(3)) + sin((k * eta_i) / sqrt(3)))) / (pow(k, 3) * pow(eta_i, 3));
		  };
    auto fft_wrapper = fftwWrapper(param.N);
    Eigen::VectorXd Psi = compute_field_with_scaled_fourier_modes(param.N, param.L, R, kernel, fft_wrapper);

    // Calculate \varphi^2, \dot{\varphi}^2 perturbations as a multiple of Psi.
    // See Eqn (3.17) of paper.
    // There is an extra factor of 0.5 in front since "generate_inhomogeneous_gaussian_random_field" use exp(2\Psi) ~ 1 + 2 \Psi for variance perturbation convention.
    double v = param.k_ast / (param.a1 * param.m);
    double alpha_varphi_sqr = 0.5 * (- 3 * pow(4*pow(v,2)+5, 2)) / (12*pow(v,4) + 50*pow(v,2) + 50);
    double alpha_dot_varphi_sqr = 0.5 * (25 - 20*pow(v,2)) / (12*pow(v,4) + 50*pow(v,2) + 50);
    
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
    Spectrum P_dtf = to_deriv_spectrum(param.m, P_f);
    Eigen::VectorXd varphi = generate_inhomogeneous_gaussian_random_field(param.N, param.L, alpha_varphi_sqr * Psi, P_f);
    Eigen::VectorXd dt_varphi = generate_inhomogeneous_gaussian_random_field(param.N, param.L, alpha_dot_varphi_sqr * Psi, P_dtf);
    
    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());

    // Save the comoving curvature perturbation for reference
    {
      decltype(workspace.state) R_dvec(R.size());
      ALGORITHM_NAMESPACE::copy(R.begin(), R.end(), R_dvec.begin());
      workspace.R_fft = workspace.fft_wrapper.execute_d2z(R_dvec);
    }
  };



/*!
  \brief Initialize a homogeneous field with amplitude f and time derivative dt_f.
  For testing the numerical code.
*/
inline auto homogeneous_field =
  [](const auto param, auto &workspace) {
    const long long int N = param.N;
    Eigen::VectorXd varphi = Eigen::VectorXd::Constant(N*N*N, param.f);
    Eigen::VectorXd dt_varphi = Eigen::VectorXd::Constant(N*N*N, param.dt_f);

    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
  };


/*!
  \brief Initialize a homogeneous field with amplitude f, plus scale-invariant perturbations (resembling quantum fluctutations).
*/
inline auto homogeneous_field_with_fluctuations =
  [](const auto param, auto &workspace) {
    const long long int N = param.N;
    Eigen::VectorXd varphi = Eigen::VectorXd::Constant(N*N*N, param.f);
    Eigen::VectorXd dt_varphi = Eigen::VectorXd::Constant(N*N*N, 0.0);

    Spectrum P_delta_varphi = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.delta_varphi_std_dev, param.k_delta_varphi, -3);
    Eigen::VectorXd delta_varphi = generate_gaussian_random_field(param.N, param.L, P_delta_varphi);
    varphi += delta_varphi;
    
    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
  };


/*!
  \brief Plane wave initial condition.
  For testing the numerical code.
*/
inline auto plane_wave =
  [](const auto param, auto &workspace) {
    const long long int N = param.N;
    Eigen::VectorXd varphi(N*N*N);
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  varphi(IDX_OF(N, a, b, c)) = cos(2 * std::numbers::pi * c / N);
	}
      }
    }
    
    Eigen::VectorXd dt_varphi = Eigen::VectorXd::Constant(N*N*N, 0);

    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());
  };


/*!
  \brief Wave packet initial condition.
*/
inline auto wave_packet =
  [](const auto param, auto &workspace) {
    const long long int N = param.N;
    Eigen::VectorXd varphi(N*N*N);
    Eigen::VectorXd dt_varphi(N*N*N);
    Eigen::VectorXd Psi(N*N*N);
    
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  double dist_to_center = sqrt(std::pow(std::min((double)a, (double)std::abs(N-a)), 2) + (b - N/3) * (b - N/3) + (c - N/3) * (c - N/3)) * (param.L / param.N);
	  varphi(IDX_OF(N, a, b, c)) = exp(- dist_to_center * dist_to_center / 40.0);
	  dt_varphi(IDX_OF(N, a, b, c)) = 0;
	  //Psi(IDX_OF(N, a, b, c)) = - param.Psi_std_dev * exp( - (b - N/2) * (b - N/2) / (2 * (param.L * param.L / 3.0 / 3.0)));
	  Psi(IDX_OF(N, a, b, c)) = - param.Psi_std_dev * cos(2 * std::numbers::pi * c / N);
	}
      }
    }

    auto &state = workspace.state;
    state.resize(varphi.size() + dt_varphi.size());
    ALGORITHM_NAMESPACE::copy(varphi.begin(), varphi.end(), state.begin());
    ALGORITHM_NAMESPACE::copy(dt_varphi.begin(), dt_varphi.end(), state.begin() + varphi.size());

    workspace.Psi.resize(Psi.size());
    ALGORITHM_NAMESPACE::copy(Psi.begin(), Psi.end(), workspace.Psi.begin());
  };


#endif
