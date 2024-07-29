/*! 
  \file random_field.hpp
  \author Siyang Ling
  \brief Utilities for generating Gaussian random fields of given spectrum and inhomogeneity.

  This file contains utilities for generating Gaussian random fields (GRF), 
  including some example spectra and a function for generating field realizations from a spectra.
  See function generate_inhomogeneous_gaussian_random_field for details.
*/
#ifndef RANDOM_FIELD_HPP
#define RANDOM_FIELD_HPP

#include "Eigen/Dense"

#include <functional>
#include <random>
#include <vector>


// A self-initializing random number generator for standard normal distribution
namespace RandomNormal
{
  void set_generator_seed(std::mt19937::result_type seed);
  std::mt19937 get_generator_from_device();
  double generate_random_normal();
}

/*! 
  \brief Typedef for spectrum \f$ P(k) \f$. Given momentum \f$ k \f$, the spectrum should return \f$ P(k) \f$.
*/
typedef std::function<double(const double)> Spectrum;

// Typical spectra.

/*!
  \brief \f$ k^\alpha \f$ power law spectrum with a sharp cutoff at \f$ k_\ast \f$.
  \param N Number of lattice points.
  \param L Box size.
  \param sigma Standard deviation \f$ \sigma \f$ of generated function \f$ f \f$.
  \param k_ast Cutoff \f$ k_\ast \f$.
  \param alpha Power law index \f$ \alpha \f$.
  \return The spectrum \f$ P \f$, which can be called to get \f$ P(k) \f$.

  The spectrum is given by
  \f{eqnarray*}{
  P(0) &=& 0 \\
  P(k) &=& P(k_0) (k/k_0)^\alpha \textrm{ for } k < k_0 \\
  \overline{f^2} &=& \sigma^2
  \f}
*/
Spectrum power_law_with_cutoff_given_amplitude_3d(const long long int N, const double L, const double sigma, const double k_ast, const double alpha);

/*!
  \brief Broken power law spectrum with the break at \f$ k_\ast \f$.
  \param N Number of lattice points.
  \param L Box size.
  \param sigma Standard deviation \f$ \sigma \f$ of generated function \f$ f \f$.
  \param k_ast The break \f$ k_\ast \f$.
  \param alpha Power law index \f$ \alpha \f$.
  \param beta Power law index \f$ \beta \f$.
  \return The spectrum \f$ P \f$, which can be called to get \f$ P(k) \f$.

  The spectrum is given by
  \f{eqnarray*}{
  P(0) &=& 0 \\
  P(k) &=& P(k_0) (k/k_0)^\alpha \textrm{ for } k < k_0 \\
  P(k) &=& P(k_0) (k/k_0)^\beta \textrm{ for } k > k_0 \\
  \overline{f^2} &=& \sigma^2
  \f}
*/
Spectrum broken_power_law_given_amplitude_3d(const long long int N, const double L, const double sigma, const double k_ast, const double alpha, const double beta);

/*!
  \brief \f$ k^\alpha \f$ power law spectrum with a sharp cutoff at \f$ k_\ast \f$.
  \param N Number of lattice points.
  \param L Box size.
  \param As The height of the spectrum \f$ A_s \f$.
  \return The spectrum \f$ P \f$, which can be called to get \f$ P(k) \f$.

  The spectrum is given by
  \f{eqnarray*}{
  P(0) &=& 0 \\
  P(k) &=& A_s
  \f}
*/
Spectrum scale_invariant_spectrum_3d(const long long int N, const double L, const double As);

/*!
  \brief Given spectrum \f$ P_\varphi \f$, return a new spectrum given by \f$ P_{\dot\varphi}(k) = (k^2 + m^2) P_\varphi(k) \f$.
 */
Spectrum to_deriv_spectrum(const double m, const Spectrum &P_f);

/*!
  \brief Given spectrum \f$ P_\varphi \f$, return a new spectrum given by \f$ P_{\dot\varphi}(k) = (k^2/a^2 + m^2) P_\varphi(k) \f$.
 */
Spectrum to_deriv_spectrum(const double m, const double a, const Spectrum &P_f);

/*!
  \brief Special case of generate_inhomogeneous_gaussian_random_field.
*/
Eigen::VectorXd generate_gaussian_random_field(const long long int N, const double L, const Spectrum &P);

/*! 
  \brief Generate an inhomogeneous 3D real Gaussian random field from spectral data P(k).
  \param N Number of lattice points.
  \param L Box size.
  \param Psi The inhomogeneity function \f$ \psi \f$, given in terms of values on the lattice (of size \f$ N^3 \f$).
  \param P The spectrum \f$ P \f$.
  \return The generated GRF, as values on the lattice (of size \f$ N^3 \f$).

  Generate an inhomogeneous Gaussian random field \f$ f \f$, such that the spectrum of \f$ f \f$ is \f$ P \f$, 
  and the variance of the field has inhomogeneity like \f$ \langle f^2(x) \rangle \approx \overline{f^2} e^{2 \psi(x)} \f$.
  See section 3.2 of paper for details of this procedure.
*/
Eigen::VectorXd generate_inhomogeneous_gaussian_random_field(const long long int N, const double L, const Eigen::VectorXd &Psi, const Spectrum &P);




#endif
