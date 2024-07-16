/*! 
  \file random_field.hpp
  \brief Utilities for generating Gaussian random fields of given spectrum and inhomogeneity.
*/
#ifndef RANDOM_FIELD_HPP
#define RANDOM_FIELD_HPP

#include <cstdlib>
#include <cassert>
#include <numbers>
#include <iostream>
#include <vector>
#include <complex>
#include <random>


#include "utility.hpp"
#include "fdm3d.hpp"


// A self-initializing random number generator for standard normal distribution
namespace RandomNormal
{
  void set_generator_seed(std::mt19937::result_type seed);
  std::mt19937 get_generator_from_device();
  double generate_random_normal();
}

// Typedef for spectrum P(k).
typedef std::function<double(const double)> Spectrum;

/*
  Specify typical spectrums.
*/

/*
  P(0) = 0.
  P(k) = P_k0 * (k/k0)^alpha for k < k0.
  <f^2> = sigma^2.
*/
Spectrum power_law_with_cutoff_given_amplitude_3d(const long long int N, const double L, const double sigma, const double k_ast, const double alpha);

/*
  P(0) = 0.
  P(k) = P_k0 * (k/k0)^alpha for k < k0.
  P(k) = P_k0 * (k/k0)^beta for k < k0.
  <f^2> = sigma^2.
*/
Spectrum broken_power_law_given_amplitude_3d(const long long int N, const double L, const double sigma, const double k_ast, const double alpha, const double beta);

/*
  P(0) = 0.
  P(k) = As.
*/
Spectrum scale_invariant_spectrum_3d(const long long int N, const double L, const double As);

// Given spectrum P_f, return P_dtf(k) = (k^2 + m^2) P_f(k).
Spectrum to_deriv_spectrum(const double m, const Spectrum &P_f);

// Given spectrum P_f, return P_dtf(k) = (k^2/a^2 + m^2) P_f(k).
Spectrum to_deriv_spectrum(const double m, const double a, const Spectrum &P_f);

// A stud spectrum for testing.
Spectrum test_spectrum(const long long int N, const double L, const double sigma, const double k_ast, const double alpha);


// Generate a homogeneous 3D real Gaussian random field phi from spectra data P(k).
Eigen::VectorXd generate_gaussian_random_field(const long long int N, const double L, const Spectrum &P);

// Generate an inhomogeneous 3D real Gaussian random field phi from spectra data P(k).
Eigen::VectorXd generate_inhomogeneous_gaussian_random_field(const long long int N, const double L, const Eigen::VectorXd &Psi, const Spectrum &P);




#endif
