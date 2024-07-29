/*! 
  \file fdm3d.hpp
  \brief Commonly used procedures for manipulating / summarizing field configuration on a 3D lattice.
*/
#ifndef FDM3D_HPP
#define FDM3D_HPP

#include "Eigen/Dense"
#include "dispatcher.hpp"


/*! 
  \brief Give the linear index of a lattice point.
*/
#define IDX_OF(N, i, j, k) ((N)*(N)*(i) + (N)*(j) + (k))

/*! 
  \brief Give the index of a lattice point, assuming that the array is in FFTW padded format.

  See <https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html>.
*/
#define PADDED_IDX_OF(N, i, j, k) ((N)*2*((N)/2+1)*(i) + 2*((N)/2+1)*(j) + (k))


Eigen::VectorXd compute_power_spectrum(const long long int N,
				       Eigen::VectorXd &f,
				       fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

Eigen::VectorXd compute_mode_power_spectrum(const long long int N, const double L, const double m,
					    Eigen::VectorXd &state,
					    fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

Eigen::VectorXd compute_inverse_laplacian(const long long int N, const double L,
					  Eigen::VectorXd &f,
					  fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

Eigen::VectorXd compute_field_with_scaled_fourier_modes(const long long int N, const double L,
							Eigen::VectorXd &f,
							std::function<double(const double)> kernel,
							fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

Eigen::VectorXd compute_cutoff_fouriers(const long long int N, const long long int M,
					Eigen::VectorXd &fft);


// Deprecated
// Eigen::VectorXd compute_power_spectrum(const long long int N, Eigen::VectorXd &phi);
// Eigen::VectorXd compute_fourier(const long long int N, const double L, Eigen::VectorXd &phi);
// Eigen::VectorXd compute_laplacian(const long long int N, const double L, const Eigen::VectorXd &f);



#endif
