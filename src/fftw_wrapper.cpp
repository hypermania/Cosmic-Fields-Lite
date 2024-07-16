#include "fftw_wrapper.hpp"


fftwWrapper::fftwWrapper(int N_) : N(N_)
{
  fftw_complex *complex_array = fftw_alloc_complex(N * N * (N / 2 + 1));
  double *real_array = fftw_alloc_real(N * N * N);
  plan_d2z = fftw_plan_dft_r2c_3d(N, N, N, real_array, complex_array, FFTW_MEASURE);
  plan_z2d = fftw_plan_dft_c2r_3d(N, N, N, complex_array, real_array, FFTW_MEASURE);
  plan_inplace_z2d = fftw_plan_dft_c2r_3d(N, N, N, complex_array, reinterpret_cast<double *>(complex_array), FFTW_MEASURE);
  fftw_free(complex_array);
  fftw_free(real_array);
}


fftwWrapper::~fftwWrapper()
{
  fftw_destroy_plan(plan_z2d);
  fftw_destroy_plan(plan_d2z);
  fftw_destroy_plan(plan_inplace_z2d);
}


Eigen::VectorXd fftwWrapper::execute_d2z(Eigen::VectorXd &in)
{
  Eigen::VectorXd out(N * N * (N / 2 + 1) * 2);
  
  // fftw_plan plan = fftw_plan_dft_r2c_3d(N, N, N, in.data(), reinterpret_cast<fftw_complex *>(out.data()), FFTW_ESTIMATE);
  // fftw_execute(plan);
  // fftw_destroy_plan(plan);
  
  fftw_execute_dft_r2c(plan_d2z, in.data(), reinterpret_cast<fftw_complex *>(out.data()));
  return out;
}


// This function destroys the input.
// See FFTW's documentation (https://www.fftw.org/fftw3_doc/Planner-Flags.html).
Eigen::VectorXd fftwWrapper::execute_z2d(Eigen::VectorXd &in)
{
  Eigen::VectorXd out(N * N * N);
  
  // fftw_plan plan = fftw_plan_dft_c2r_3d(N, N, N, reinterpret_cast<fftw_complex *>(in.data()), out.data(), FFTW_ESTIMATE);
  // fftw_execute(plan);
  // fftw_destroy_plan(plan);
  
  fftw_execute_dft_c2r(plan_z2d, reinterpret_cast<fftw_complex *>(in.data()), out.data());
  return out;
}


// Memory allocation / deallocation seems to take a lot of time.
// This version is useful if we want to allocate once and reuse the same memory again and again.
void fftwWrapper::execute_z2d(Eigen::VectorXd &in, Eigen::VectorXd &out)
{
  fftw_execute_dft_c2r(plan_z2d, reinterpret_cast<fftw_complex *>(in.data()), out.data());
}


// Note: The output data format for this function has extra padding. Use PADDED_IDX_OF to access the output.
// This was mentioned in FFTW's documentation (https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html).
void fftwWrapper::execute_inplace_z2d(Eigen::VectorXd &inout)
{
  // Eigen::VectorXd result(N * N * N);
  // fftw_execute_dft_c2r(plan_z2d, reinterpret_cast<fftw_complex *>(inout.data()), result.data());
  // inout.head(N * N * N) = result;
  fftw_execute_dft_c2r(plan_inplace_z2d, reinterpret_cast<fftw_complex *>(inout.data()), inout.data());
}

Eigen::VectorXd fftwWrapper::execute_batched_d2z(Eigen::VectorXd &in)
{
  Eigen::VectorXd out(N * N * (N / 2 + 1) * 2 * 2);
  fftw_execute_dft_r2c(plan_d2z, in.data(), reinterpret_cast<fftw_complex *>(out.data()));
  fftw_execute_dft_r2c(plan_d2z, in.data() + N*N*N, reinterpret_cast<fftw_complex *>(out.data()) + N*N*(N/2+1));
  return out;
}
