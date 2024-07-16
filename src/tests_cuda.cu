#include "tests_cuda.cuh"

void print_cuda_info(void)
{
  std::cout << "==== Printing CUDA information ====" << '\n';
  
  int driver_version, runtime_version;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  std::cout << "CUDA driver_version = " << driver_version << '\n';
  std::cout << "CUDA runtime_version = " << runtime_version << '\n';

  int device_count, device_num;
  cudaGetDeviceCount(&device_count);
  cudaGetDevice(&device_num);
  std::cout << "CUDA device_count = " << device_count << '\n';
  std::cout << "CUDA device_num = " << device_num << '\n';

  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, device_num);
  std::cout << "CUDA device name = " << device_properties.name << '\n';
  std::cout << "CUDA global memory size = " << device_properties.totalGlobalMem << '\n';
  std::cout << "CUDA shared memory size (per block) = " << device_properties.sharedMemPerBlock << '\n';
  std::cout << "CUDA regsPerBlock = " << device_properties.regsPerBlock << '\n';
  std::cout << "CUDA warpSize = " << device_properties.warpSize << '\n';
  std::cout << "CUDA maxThreadsPerBlock = " << device_properties.maxThreadsPerBlock << '\n';
  std::cout << "CUDA maxThreadsDim = "
	    << device_properties.maxThreadsDim[0] << ","
	    << device_properties.maxThreadsDim[1] << ","
	    << device_properties.maxThreadsDim[2] << '\n';
  std::cout << "CUDA maxGridSize = "
	    << device_properties.maxGridSize[0] << ","
	    << device_properties.maxGridSize[1] << ","
	    << device_properties.maxGridSize[2] << "," << '\n';
  std::cout << "CUDA clockRate = " << device_properties.clockRate << '\n';
  std::cout << "CUDA multiProcessorCount = " << device_properties.multiProcessorCount << '\n';
  std::cout << "CUDA memoryClockRate = " << device_properties.memoryClockRate << '\n';
  std::cout << "CUDA memoryBusWidth = " << device_properties.memoryBusWidth << '\n';
  
  std::cout << "===================================" << '\n';
}

/*
void fill_dxdt_radiation_3d_cuda_impl(const long long int N, const double m,
				      const double a_t, const double H_t, const double inv_ah_sqr,
				      const thrust::device_vector<double> &x,
				      thrust::device_vector<double> &dxdt)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(16, 16);
  compute_deriv_radiation<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, a_t, H_t, inv_ah_sqr, N);
}
*/

