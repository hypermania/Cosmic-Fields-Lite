/*! 
  \file fdm3d.hpp
  \author Siyang Ling
  \brief Common procedures for manipulating / summarizing field configuration on a 3D lattice.
*/
#ifndef FDM3D_HPP
#define FDM3D_HPP

#include "Eigen/Dense"
#include "dispatcher.hpp"


/*! 
  \brief Give the index of a lattice point, assuming row major ordering in (i,j,k).
*/
#define IDX_OF(N, i, j, k) ((N)*(N)*(i) + (N)*(j) + (k))


/*! 
  \brief Give the index of a lattice point, assuming that the array is in FFTW padded format.

  See <https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html> for details of the format.
*/
#define PADDED_IDX_OF(N, i, j, k) ((N)*2*((N)/2+1)*(i) + 2*((N)/2+1)*(j) + (k))


/*! 
  \brief Sum Fourier mode power of a field over directions.
  \param N Number of lattice points.
  \param f The field on a 3D lattice. Should be a vector of size \f$ N^3 \f$ with row major ordering. See IDX_OF.
  \param fft_wrapper A fftwWrapper initialized to do Fourier transforms on grid size \f$ N \f$.
  \return A vector of size \f$ 3 (N/2)^2 + 1\f$, with its \f$ s \f$ index containing the power in Fourier modes with wavenumber \f$ \sqrt{s} k_\mathrm{IR} \f$.
  Specifically:
  \f{eqnarray*}{
  \mathrm{output}[s] &=& \sum_{i^2+j^2+k^2=s} |\tilde{f}_{i,j,k}|^2 \\
  \tilde{f}_{i,j,k} &=& \sum_{a,b,c} e^{-2 \pi i (a,b,c).(i,j,k) / N} f_{a,b,c}
  \f}
  Here, \f$ \tilde{f} \f$ is the DFT of \f$ f \f$, \f$ (i,j,k) \f$ labels a site on the reciprocal lattice, and \f$ -N/2 + 1 \leq i,j,k \leq N/2 \f$.
  See <https://garrettgoon.com/gaussian-fields/> for details on this convention.
*/
Eigen::VectorXd compute_power_spectrum(const long long int N,
				       Eigen::VectorXd &f,
				       fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

/*! 
  \brief Sum Fourier mode power of a field over directions, along with the power in time derivatives.
  \param N Number of lattice points.
  \param L Box size.
  \param m Mass \f$ m \f$ of (free) scalar field.
  \param a_t Current scale factor.
  \param state The state \f$ (\varphi, \dot{\varphi}) \f$ of a scalar field on a 3D lattice. 
  Should be a vector of size \f$ 2 N^3 \f$, with the first half (first \f$ N^3 \f$ indices) containing \f$ \varphi \f$, and the second half containing \f$ \dot{\varphi} \f$.
  \param fft_wrapper A fftwWrapper initialized to do Fourier transforms on grid size \f$ N \f$.
  \return A vector of size \f$ 3 (N/2)^2 + 1\f$, with its \f$ s \f$ index containing the power in Fourier modes with wavenumber \f$ \sqrt{s} k_\mathrm{IR} \f$.
  Specifically:
  \f{eqnarray*}{
  \mathrm{output}[s] &=& \sum_{i^2+j^2+k^2=s} |\tilde{\varphi}_{i,j,k}|^2 + \frac{|\dot{\tilde{\varphi}}_{i,j,k}|^2}{\omega_k^2} \\
  \omega_k^2 &=& m^2 + s k_\mathrm{IR}^2 / a^2(t)
  \f}
  Here, \f$ \tilde{\varphi}_{a,b,c} \f$ and \f$ \dot{\tilde{\varphi}}_{a,b,c} \f$ are the DFT's,
  \f$ (i,j,k) \f$ labels a site on the reciprocal lattice, and \f$ -N/2 + 1 \leq i,j,k \leq N/2 \f$;
  see <https://garrettgoon.com/gaussian-fields/> for details on this convention.
  Also see compute_power_spectrum.
*/
Eigen::VectorXd compute_mode_power_spectrum(const long long int N, const double L, const double m, const double a_t,
					    Eigen::VectorXd &state,
					    fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

// Eigen::VectorXd compute_mode_power_spectrum(const long long int N, const double L, const double m,
// 					    Eigen::VectorXd &state,
// 					    fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

/*! 
  \brief Compute the inverse Laplacian of a field. AKA solve the Poisson equation.
  \param N Number of lattice points.
  \param L Box size.
  \param f The field on a 3D lattice. Should be a vector of size \f$ N^3 \f$ with row major ordering. See IDX_OF.
  \param fft_wrapper A fftwWrapper initialized to do Fourier transforms on grid size \f$ N \f$.
  \return The solution to the Poisson equation with RHS \f$ f \f$, namely \f$ \nabla^{-2} f \f$.
  The output have zero homogeneous mode regardless of whether \f$ f \f$ has one.
*/
Eigen::VectorXd compute_inverse_laplacian(const long long int N, const double L,
					  Eigen::VectorXd &f,
					  fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);


/*! 
  \brief Scale each Fourier mode of a field by a kernel, returning the new field.
  \param N Number of lattice points.
  \param L Box size.
  \param f The field on a 3D lattice. Should be a vector of size \f$ N^3 \f$ with row major ordering. See IDX_OF.
  \param kernel A function \f$ K \f$ determining how the Fourier modes are scaled.
  \param fft_wrapper A fftwWrapper initialized to do Fourier transforms on grid size \f$ N \f$.
  \return The field with \f$ f_{\bf k} \mapsto K(k) f_{\bf k} \f$, where \f$ K \f$ is given by kernel.
*/
Eigen::VectorXd compute_field_with_scaled_fourier_modes(const long long int N, const double L,
							Eigen::VectorXd &f,
							std::function<double(const double)> kernel,
							fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);


/*! 
  \brief Downsample a Fourier transform on a \f$ N^3 \f$ grid so that it looks like a Fourier transform on a \f$ M^3 \f$ grid.
  \param N Number of lattice points (of full grid).
  \param M Number of lattice points (of downsampled grid).
  \param fft The DFT of a real field. Should be a vector of size \f$ 2N^2(N/2+1) \f$.
  \return The downsampled DFT the input DFT. Should be a vector of size \f$ 2M^2(M/2+1) \f$.
*/
Eigen::VectorXd compute_cutoff_fouriers(const long long int N, const long long int M,
					Eigen::VectorXd &fft);


// Deprecated
// Eigen::VectorXd compute_power_spectrum(const long long int N, Eigen::VectorXd &phi);
// Eigen::VectorXd compute_fourier(const long long int N, const double L, Eigen::VectorXd &phi);
// Eigen::VectorXd compute_laplacian(const long long int N, const double L, const Eigen::VectorXd &f);



#endif
