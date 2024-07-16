#ifndef FFTW_WRAPPER_HPP
#define FFTW_WRAPPER_HPP

#include <iostream>

#include <Eigen/Dense>
#include <fftw3.h>


/*
  Wrapper for various fftw functions for a N*N*N grid.
*/
struct fftwWrapper {
  int N;
  fftw_plan plan_d2z;
  fftw_plan plan_z2d;
  fftw_plan plan_inplace_z2d;
  explicit fftwWrapper(int N_);
  ~fftwWrapper();
  
  Eigen::VectorXd execute_d2z(Eigen::VectorXd &in);
  Eigen::VectorXd execute_batched_d2z(Eigen::VectorXd &in);
  Eigen::VectorXd execute_z2d(Eigen::VectorXd &in);
  void execute_z2d(Eigen::VectorXd &in, Eigen::VectorXd &out);
  void execute_inplace_z2d(Eigen::VectorXd &inout);
  
  fftwWrapper(const fftwWrapper &) = delete;
  fftwWrapper &operator=(const fftwWrapper &) = delete;
  fftwWrapper(fftwWrapper &&) = delete;
  fftwWrapper &operator=(fftwWrapper &&) = delete;
};


#endif
