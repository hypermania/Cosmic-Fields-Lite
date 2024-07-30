/*!
  \file eigen_wrapper.hpp
  \author Siyang Ling
  \brief Wrap some Eigen functionalites.
*/
#ifndef EIGEN_WRAPPER_HPP
#define EIGEN_WRAPPER_HPP

#include <Eigen/Dense>

void copy_vector(Eigen::VectorXd &out, const Eigen::VectorXd &in);

#endif
