/*!
  \file dispatcher.hpp
  \author Siyang Ling
  \brief Automatically dispatching between using FFTW and CUFFT libraries.
*/
#ifndef DISPATCHER_HPP
#define DISPATCHER_HPP

#include "fftw_wrapper.hpp"

#ifndef DISABLE_CUDA
#include <thrust/device_vector.h>
#include "cuda_wrapper.cuh"
#define ALGORITHM_NAMESPACE thrust
#else
#define ALGORITHM_NAMESPACE std
#endif


// An empty placeholder object
struct empty {};

// Dispatcher for fftWrapper* types
template<typename Vector>
struct fftWrapperDispatcher {
  typedef empty D2Z;
  typedef empty BatchedD2Z;
  typedef empty Generic;
};

#ifndef DISABLE_CUDA
template<>
struct fftWrapperDispatcher<thrust::device_vector<double>> {
  typedef cufftWrapperD2Z D2Z;
  typedef cufftWrapperBatchedD2Z BatchedD2Z;
  //typedef cufftWrapper Generic;
  typedef cufftWrapperNoBatching Generic;
};
#endif

template<>
struct fftWrapperDispatcher<Eigen::VectorXd> {
  typedef empty D2Z;
  typedef empty BatchedD2Z;
  typedef fftwWrapper Generic;
};


#endif
