/*! 
  \file fftw_wrapper.hpp
  \author Siyang Ling
  \brief Wrapper for FFTW library.
*/
#ifndef FFTW_WRAPPER_HPP
#define FFTW_WRAPPER_HPP

#include <iostream>

#include <Eigen/Dense>
#include <fftw3.h>


/*!
  \brief Wrapper for various FFTW functions for a \f$ N^3 \f$ grid.

  See <https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html> for details.
*/
struct fftwWrapper {
  int N;
  fftw_plan plan_d2z;
  fftw_plan plan_z2d;
  fftw_plan plan_inplace_z2d;
  explicit fftwWrapper(int N_); /*!< Constructor for grid size \f$ N \f$. */
  ~fftwWrapper();

  /*!
    \brief (Double floating point) Real to complex transform.
    \param in A real vector of size \f$ N^3 \f$.
    \return A real vector of size \f$ 2 N^2 (N/2+1) \f$ (or a complex vector of size \f$ N^2 (N/2+1) \f$), containing the discrete Fourier transform of input.
  */
  Eigen::VectorXd execute_d2z(Eigen::VectorXd &in);
  
  /*!
    \brief (Double floating point) Real to complex transform.
    \param in A real vector of size \f$ 2 N^3 \f$.
    \return A real vector of size \f$ 4 N^2 (N/2+1) \f$ (or a complex vector of size \f$ 2 N^2 (N/2+1) \f$), containing discrete Fourier transforms of input. The first \f$ 2 N^2 (N/2+1) \f$ entries of the output are the DFT of the first \f$ N^3 \f$ entries of the input, and similar for the rest.
  */
  Eigen::VectorXd execute_batched_d2z(Eigen::VectorXd &in);

  /*!
    \brief (Double floating point) Complex to real transform.
    \param in A real vector of size \f$ 2 N^2 (N/2+1) \f$ (or a complex vector of size \f$ N^2 (N/2+1) \f$).
    \return A real vector of size \f$ N^3 \f$, containing the inverse discrete Fourier transform of input.
    \note This function destroys the information in the input `in`.
    See FFTW's documentation <https://www.fftw.org/fftw3_doc/Planner-Flags.html>.
  */
  Eigen::VectorXd execute_z2d(Eigen::VectorXd &in);

  /*! 
    \brief No-return version of the complex to real transform.
    \note This version is useful if you want to reuse the same memory location for the output; doing this can reduce unnecessary memory allocation / deallocation, saving lots of time.
    Like the other version, this function destroys the data in input `in`.
  */
  void execute_z2d(Eigen::VectorXd &in, Eigen::VectorXd &out);

  /*!
    \brief In-place version of the complex to real transform.
    \param inout A real vector of size \f$ 2 N^2 (N/2+1) \f$ (or a complex vector of size \f$ N^2 (N/2+1) \f$). After the function call the data in `inout` is changed its inverse DFT. The vector still has size \f$ 2 N^2 (N/2+1) \f$, but only \f$ N^3 \f$ of the entries are meaningful. The entries are in FFTW padded format.
    \note Make sure to access the elements inside with PADDED_IDX_OF (instead of IDX_OF).
    See <https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html> for details of the padded format.
  */
  void execute_inplace_z2d(Eigen::VectorXd &inout);
  
  fftwWrapper(const fftwWrapper &) = delete;
  fftwWrapper &operator=(const fftwWrapper &) = delete;
  fftwWrapper(fftwWrapper &&) = delete;
  fftwWrapper &operator=(fftwWrapper &&) = delete;
};


#endif
