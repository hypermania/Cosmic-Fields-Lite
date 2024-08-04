/*! 
  \file observer.hpp
  \author Siyang Ling
  \brief Implements "observers", which controls what gets saved during simulations.
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
  double t_start;
  double t_end;
  double t_interval;
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
