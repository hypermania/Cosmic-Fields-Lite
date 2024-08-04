/*! 
  \file cuda_wrapper.cuh
  \author Siyang Ling
  \brief Wrapper for CUDA Toolkit.
*/
#ifndef CUDA_WRAPPER_CUH
#define CUDA_WRAPPER_CUH

#include <iostream>

#include <Eigen/Dense>

#include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// #include <thrust/fill.h>
// #include <thrust/transform.h>

#include "cufft.h"
#include "cufftXt.h"
#include <cuda_runtime.h>



typedef decltype(Eigen::VectorXd().begin()) eigen_iterator;
typedef decltype(thrust::device_vector<double>().begin()) thrust_iterator;
typedef thrust::detail::normal_iterator<thrust::device_ptr<const double>> thrust_const_iterator;
typedef Eigen::internal::pointer_based_stl_iterator<Eigen::Matrix<double, -1, 1>> eigen_iterator_2;


/*
  Explicit template instantiation declarations for the thrust library.
  They are declared here so that they are instantiatiated in cuda_wrapper.cu (and compiled with nvcc),
  and don't get instantiated in other translation units.
  This is necessary since we want to call thrust functions in translation units compiled by other compilers (g++ / icpx).
*/
extern template class thrust::device_vector<double>;
extern template class thrust::device_ptr<double>;
extern template thrust::device_ptr<double> thrust::for_each_n(const thrust::detail::execution_policy_base<thrust::cuda_cub::tag> &, thrust::device_ptr<double>, unsigned long, thrust::detail::device_generate_functor<thrust::detail::fill_functor<double>>);
extern template eigen_iterator thrust::copy(const thrust::detail::execution_policy_base<thrust::cuda_cub::cross_system<thrust::cuda_cub::tag, thrust::system::cpp::detail::tag>> &, thrust_const_iterator, thrust_const_iterator, eigen_iterator);

extern template thrust_iterator thrust::copy(eigen_iterator, eigen_iterator, thrust_iterator);
extern template eigen_iterator thrust::copy(thrust_iterator, thrust_iterator, eigen_iterator);

//Eigen::VectorXd copy_vector(const thrust::device_vector<double> &in);
void copy_vector(Eigen::VectorXd &out, const thrust::device_vector<double> &in);
//void copy_vector(Eigen::VectorXd &out, const Eigen::VectorXd &in);


void show_gpu_memory_usage(void);

// Wrapper for 3D cufftPlan3d. Performs double to complex double FFT for a \f$ N^3 \f$ grid.
struct cufftWrapperD2Z {
  int N;
  cufftHandle plan;
  explicit cufftWrapperD2Z(int N_);
  ~cufftWrapperD2Z();
  thrust::device_vector<double> execute(thrust::device_vector<double> &in);
  
  cufftWrapperD2Z(const cufftWrapperD2Z &) = delete;
  cufftWrapperD2Z &operator=(const cufftWrapperD2Z &) = delete;
  cufftWrapperD2Z(cufftWrapperD2Z &&) = delete;
  cufftWrapperD2Z &operator=(cufftWrapperD2Z &&) = delete;
};


// Wrapper for 3D cufftPlanMany. Performs two double to complex double FFT for a \f$ N^3 \f$ grid.
struct cufftWrapperBatchedD2Z {
  int N;
  cufftHandle plan;
  explicit cufftWrapperBatchedD2Z(int N_);
  ~cufftWrapperBatchedD2Z();
  thrust::device_vector<double> execute(thrust::device_vector<double> &in);
  
  cufftWrapperBatchedD2Z(const cufftWrapperBatchedD2Z &) = delete;
  cufftWrapperBatchedD2Z &operator=(const cufftWrapperBatchedD2Z &) = delete;
  cufftWrapperBatchedD2Z(cufftWrapperBatchedD2Z &&) = delete;
  cufftWrapperBatchedD2Z &operator=(cufftWrapperBatchedD2Z &&) = delete;
};

/*!
  \brief Wrapper for various cufft functions for a \f$ N^3 \f$ grid. Similar to fftwWrapper.

  See <https://docs.nvidia.com/cuda/cufft/index.html>.
*/
struct cufftWrapper {
  int N;
  cufftHandle plan_d2z;
  cufftHandle plan_batched_d2z;
  cufftHandle plan_z2d;
  thrust::device_vector<double> work_area;
  explicit cufftWrapper(int N_);
  ~cufftWrapper();
  
  thrust::device_vector<double> execute_d2z(thrust::device_vector<double> &in);
  thrust::device_vector<double> execute_batched_d2z(thrust::device_vector<double> &in);
  thrust::device_vector<double> execute_z2d(thrust::device_vector<double> &in);
  
  cufftWrapper(const cufftWrapper &) = delete;
  cufftWrapper &operator=(const cufftWrapper &) = delete;
  cufftWrapper(cufftWrapper &&) = delete;
  cufftWrapper &operator=(cufftWrapper &&) = delete;
};

/*!
  \brief Wrapper for various cufft functions for a \f$ N^3 \f$ grid. Similar to fftwWrapper.

  Uses less GPU memory than cufftWrapper.
  See <https://docs.nvidia.com/cuda/cufft/index.html>.
*/
struct cufftWrapperNoBatching {
  int N;
  cufftHandle plan_d2z;
  cufftHandle plan_z2d;
  thrust::device_vector<double> work_area;
  explicit cufftWrapperNoBatching(int N_);
  ~cufftWrapperNoBatching();
  
  thrust::device_vector<double> execute_d2z(thrust::device_vector<double> &in);
  thrust::device_vector<double> execute_batched_d2z(thrust::device_vector<double> &in);
  thrust::device_vector<double> execute_z2d(thrust::device_vector<double> &in);
  void execute_inplace_z2d(thrust::device_vector<double> &inout);
  
  cufftWrapperNoBatching(const cufftWrapperNoBatching &) = delete;
  cufftWrapperNoBatching &operator=(const cufftWrapperNoBatching &) = delete;
  cufftWrapperNoBatching(cufftWrapperNoBatching &&) = delete;
  cufftWrapperNoBatching &operator=(cufftWrapperNoBatching &&) = delete;
};


#endif
