/*! 
  \file equations_cuda.cuh
  \brief Header for field equations that runs on the GPU.
*/
#ifndef EQUATIONS_CUDA_CUH
#define EQUATIONS_CUDA_CUH

#include "equations.hpp"

#include <thrust/device_vector.h>

#include "odeint_thrust/thrust.hpp"

struct CudaKleinGordonEquationInFRW {
  typedef thrust::device_vector<double> Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<Vector> Workspace;
  Workspace &workspace;
  
  CudaKleinGordonEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(const Workspace &workspace, const double t);
  static Vector compute_dot_energy_density(const Workspace &workspace, const double t);
};


struct CudaLambdaEquationInFRW {
  typedef thrust::device_vector<double> Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<Vector> Workspace;
  Workspace &workspace;
  
  CudaLambdaEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


struct CudaSqrtPotentialEquationInFRW {
  typedef thrust::device_vector<double> Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<Vector> Workspace;
  Workspace &workspace;
  
  CudaSqrtPotentialEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


struct CudaFixedCurvatureEquationInFRW {
  typedef thrust::device_vector<double> Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<Vector> Workspace;
  Workspace &workspace;
  
  CudaFixedCurvatureEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(const Workspace &workspace, const double t);
};


struct CudaComovingCurvatureEquationInFRW {
  typedef thrust::device_vector<double> Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<Vector> Workspace;
  Workspace &workspace;
  
  CudaComovingCurvatureEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(Workspace &workspace, const double t);
};


struct CudaApproximateComovingCurvatureEquationInFRW {
  typedef thrust::device_vector<double> Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<Vector> Workspace;
  Workspace &workspace;
  
  CudaApproximateComovingCurvatureEquationInFRW(Workspace &workspace_) : workspace(workspace_) {}
  
  void operator()(const State &, State &, const double);

  static Vector compute_energy_density(Workspace &workspace, const double t);
};


// Explicit template instantiation declaration for the thrust library.
extern template double thrust::reduce(const thrust::detail::execution_policy_base<thrust::cuda_cub::tag> &, thrust_const_iterator, thrust_const_iterator, double, boost::numeric::odeint::detail::maximum<double>);

// Deprecated function for testing CUDA kernels.
/*
void compute_deriv_test(const Eigen::VectorXd &in, Eigen::VectorXd &out,
			const double m, const double lambda,
			const double a_t, const double H_t, const double inv_ah_sqr,
			const long long int N);
*/
void kernel_test(const thrust::device_vector<double> &R_fft, thrust::device_vector<double> &Psi, thrust::device_vector<double> &dPsidt,
		 const long long int N, const double L, const double m,
		 const double a_t, const double H_t, const double eta_t, const double inv_ah_sqr,
		 const double t, fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

#endif
