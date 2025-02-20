/*! 
  \file fdm3d_cuda.cuh
  \author Siyang Ling
  \brief CUDA implementation for fdm3d.hpp. Common procedures for manipulating / summarizing field configuration on a 3D lattice.
*/
#ifndef FDM3D_CUDA_CUH
#define FDM3D_CUDA_CUH

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

//#include "odeint_thrust/thrust.hpp"
#include "cuda_wrapper.cuh"
#include "dispatcher.hpp"

#include "fdm3d.hpp"

/*!
  \brief CUDA version of identically named function in fdm3d.hpp.
*/
thrust::device_vector<double> compute_mode_power_spectrum(const long long int N, const double L, const double m, const double a_t,
							  thrust::device_vector<double> &state,
							  fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

/*!
  \brief CUDA version of identically named function in fdm3d.hpp.
*/
thrust::device_vector<double> compute_power_spectrum(const long long int N,
						     thrust::device_vector<double> &f,
						     fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

thrust::device_vector<double> compute_laplacian(const long long int N, const double L,
						thrust::device_vector<double> &f);

/*!
  \brief CUDA version of identically named function in fdm3d.hpp.
*/
thrust::device_vector<double> compute_inverse_laplacian(const long long int N, const double L,
							thrust::device_vector<double> &f,
							fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

/*!
  \brief CUDA version of identically named function in fdm3d.hpp.
*/
thrust::device_vector<double> compute_cutoff_fouriers(const long long int N, const long long int M,
						      const thrust::device_vector<double> &fft);

// void compute_inverse_laplacian_test(const long long int N, const double L,
// 				    thrust::device_vector<double> &fft);
#endif
