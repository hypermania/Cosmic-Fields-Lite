/*! 
  \file observer.hpp
  \author Siyang Ling
  \brief Implements "observers", which controls what gets saved during simulations.

  Observers are used by the odeint library.
  The `operator()` function of the observer is called at each time step.
  See <https://www.boost.org/doc/libs/1_85_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/integrate_functions.html> for details on how observers are used for a simulation.
*/

#ifndef OBSERVER_HPP
#define OBSERVER_HPP

#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

#include "Eigen/Dense"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

#include "odeint_eigen/eigen_operations.hpp"

#include "eigen_wrapper.hpp"
#include "fdm3d.hpp"
#include "io.hpp"
#include "physics.hpp"
#include "workspace.hpp"

#ifndef DISABLE_CUDA
#include "cuda_wrapper.cuh"
#include "fdm3d_cuda.cuh"
#endif

/*!
  \brief An "observer" used to save spectra and slices during the simulation at roughly constant time intervals.

  \tparam Equation This type parameter is necessary for selecting the right compute_energy_density function.
  (Each system has its own way to compute energy density.)

  \tparam save_field_spectrum Control whether field spectrum should be saved. 
  The output is \f$ \abs{\varphi_{\vb{k}}}^2 + \abs{\dot{\varphi}_{\vb{k}}}^2 / \omega_k^2 \f$ summed over directions.
  Saves to file `dir/varphi_plus_spectrum_(idx).dat`, where `(idx)` is the index of save.
  See compute_mode_power_spectrum for more details.

  \tparam save_density_spectrum Control whether energy density spectrum should be saved.
  The output is \f$ \abs{\rho_{\vb{k}}}^2 \f$ summed over directions.
  Saves to file `dir/rho_spectrum_(idx).dat`, where `(idx)` is the index of save.
  See compute_power_spectrum for more details.

  \tparam save_density Control whether field spectrum should be saved.
  The output is a constant-z slice of density spectrum and the density spectrum averaged over the z axis.
  Saves to files `dir/rho_slice_(idx).dat` and `dir/rho_axis_average_(idx).dat`, where `(idx)` is the index of save.

  Saves spectra and slices to `dir` during the simulation at roughly constant time intervals.
  During the simulation, the `operator()` function is called at each time step.
  We don't want to save a snapshot at every time step, so use `t_interval` to control when to save data.

  The template parameters can be used to choose what to save.
  By default, the observer saves field and density spectra, but not the slices.
*/
template<typename Equation,
	 bool save_field_spectrum = true,
	 bool save_density_spectrum = true,
	 bool save_density = false>
struct ConstIntervalObserver {
  typedef typename Equation::Workspace Workspace;
  typedef typename Workspace::State State;
  typedef State Vector;
  Workspace &workspace;
  int idx;
  std::string dir;
  double t_start; /*!< Start time of simulation. */
  double t_end; /*!< End time of simulation. */
  double t_interval; /*!< Save to file in `dir` every `t_interval`. */
  double t_last;

  template<typename Param>
  ConstIntervalObserver(const std::string &dir_, const Param &param, Equation &eqn) :
    workspace(eqn.workspace), idx(0), dir(dir_),
    t_start(param.t_start), t_end(param.t_end), t_interval(param.t_interval), t_last(param.t_start) {}

  ConstIntervalObserver(const ConstIntervalObserver &) = default;

  void operator()(const State &x, double t)
  {
    if(t >= t_last + t_interval || t == t_end || t == t_start) {
      const long long int N = workspace.N;
      const double L = workspace.L;
      const double m = workspace.m;
      const double a_t = workspace.cosmology.a(t);
      
      if constexpr(save_field_spectrum) {
	  Vector varphi_plus_spectrum = compute_mode_power_spectrum(N, L, m, a_t, workspace.state, workspace.fft_wrapper);
	  Eigen::VectorXd varphi_plus_spectrum_out(varphi_plus_spectrum.size());
	  copy_vector(varphi_plus_spectrum_out, varphi_plus_spectrum);
	  write_VectorXd_to_filename_template(varphi_plus_spectrum_out, dir + "varphi_plus_spectrum_%d.dat", idx);
	}

      if constexpr(save_density_spectrum) {
	  Vector rho = Equation::compute_energy_density(workspace, t);
	  Vector rho_spectrum = compute_power_spectrum(N, rho, workspace.fft_wrapper);
	  Eigen::VectorXd rho_spectrum_out(rho_spectrum.size());
	  copy_vector(rho_spectrum_out, rho_spectrum);
	  write_VectorXd_to_filename_template(rho_spectrum_out, dir + "rho_spectrum_%d.dat", idx);
	}
      
      if constexpr(save_density) {
	  Vector rho = Equation::compute_energy_density(workspace, t);
	  Eigen::VectorXd rho_copy(rho.size());
	  copy_vector(rho_copy, rho);
	  Eigen::VectorXd rho_slice = rho_copy.head(N*N); // Save the density for a = 0 slice.
	  Eigen::VectorXd rho_axis_average = rho_copy.reshaped(N*N, N).rowwise().mean(); // Save the density overaged over a axis.

	  write_VectorXd_to_filename_template(rho_slice, dir + "rho_slice_%d.dat", idx);
	  write_VectorXd_to_filename_template(rho_axis_average, dir + "rho_axis_average_%d.dat", idx);
	}
      
      workspace.t_list.push_back(t);
      t_last = t;
      ++idx;
    }
  }
};



#endif
