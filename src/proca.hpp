/*
  Tools related to Proca fields.
*/
#ifndef PROCA_HPP
#define PROCA_HPP

#include "Eigen/Dense"
#include "random_field.hpp"
#include "dispatcher.hpp"

// Our convention for the state of a Proca field is a vector [Ax, Ay, Az, dt_Ax, dt_Ay, dt_Az].
// Namely, 3 copies of a Klein Gordon field.

/*! 
  \brief Generate 3 concatenated Gaussian random fields. Namely a Proca field.
  \param N Number of lattice points.
  \param L Box size.
  \param P The spectrum \f$ P \f$.
  \return The generated GRF, as values on the lattice (of size \f$ 3 N^3 \f$).

  Generate 3 Gaussian random fields \f$ [A_x, A_y, A_z] \f$, such that the spectrum of \f$ A_i \f$ is \f$ P \f$.
*/
Eigen::VectorXd generate_gaussian_random_proca_field(const long long int N, const double L, const Spectrum &P);

/*! 
  \brief Generate 3 concatenated Gaussian random fields. Namely a Proca field.
  \param N Number of lattice points.
  \param fields A Proca field. Namely a vector of length \f$ 3 N^3 \f$.
  \param fft_wrapper
  
  Modify the Proca field, so that only the transverse components are retained.
*/
void proca_project_to_transverse(const long long int N, Eigen::VectorXd &fields, fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper);

#endif
