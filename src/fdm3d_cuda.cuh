#ifndef FDM3D_CUDA_H
#define FDM3D_CUDA_H

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

//#include "odeint_thrust/thrust.hpp"
#include "cuda_wrapper.cuh"
#include "dispatcher.hpp"

#define IDX_OF(N, i, j, k) ((N)*(N)*(i) + (N)*(j) + (k))

thrust::device_vector<double> compute_mode_power_spectrum(const long long int N, const double L, const double m,
							  thrust::device_vector<double> &state,
							  fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

thrust::device_vector<double> compute_power_spectrum(const long long int N,
						     thrust::device_vector<double> &f,
						     fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

thrust::device_vector<double> compute_laplacian(const long long int N, const double L,
						thrust::device_vector<double> &f);

thrust::device_vector<double> compute_inverse_laplacian(const long long int N, const double L,
							thrust::device_vector<double> &f,
							fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper);

thrust::device_vector<double> compute_cutoff_fouriers(const long long int N, const long long int M,
						      const thrust::device_vector<double> &fft);

void compute_inverse_laplacian_test(const long long int N, const double L,
				    thrust::device_vector<double> &fft);
#endif
