#include "cuda_wrapper.cuh"



template class thrust::device_vector<double>;
template class thrust::device_ptr<double>;
template thrust::device_ptr<double> thrust::for_each_n(const thrust::detail::execution_policy_base<thrust::cuda_cub::tag> &, thrust::device_ptr<double>, unsigned long, thrust::detail::device_generate_functor<thrust::detail::fill_functor<double>>);
template eigen_iterator thrust::copy(const thrust::detail::execution_policy_base<thrust::cuda_cub::cross_system<thrust::cuda_cub::tag, thrust::system::cpp::detail::tag>> &, thrust_const_iterator, thrust_const_iterator, eigen_iterator);

template thrust_iterator thrust::copy(eigen_iterator, eigen_iterator, thrust_iterator);
template eigen_iterator thrust::copy(thrust_iterator, thrust_iterator, eigen_iterator);

/*
  This code doesn't work as I intended it to.
  I want to be able to call this function in translation units compiled by g++/icpx.
  However, whenever I call a Eigen::VectorXd constructor compiled by nvcc and call its destructor compiled by g++/icpx, I get a segfault.
  Calling the constructor/destructor compiled by the same compiler is okay though.
*/
Eigen::VectorXd copy_vector(const thrust::device_vector<double> &in)
{
  //Eigen::VectorXd out(static_cast<long long int>(in.size()));
  //Eigen::VectorXd out = Eigen::VectorXd::Zero(in.size());
  Eigen::VectorXd out;
  out.resize(static_cast<long long int>(in.size()));
  out.array() = 0;
  std::cout << "out.size() = " << out.size() << '\n';
  cudaMemcpy((void *)out.data(), (const void *)thrust::raw_pointer_cast(in.data()), in.size() * sizeof(double), cudaMemcpyDeviceToHost);
  // std::cout << "error = " << error << '\n';
  // std::cout << "in.size() = " << in.size() << '\n';
  return out;
}

/*
  Same issue as above.
*/
Eigen::VectorXd copy_vector(const Eigen::VectorXd &in)
{
  return Eigen::VectorXd(in);
}

/*
  This code works, but now Eigen::VectorXd &out has to be allocated outside this function.
  (Typically in a translation unit compiled by g++/icpx.)
*/
void copy_vector(Eigen::VectorXd &out, const thrust::device_vector<double> &in)
{
  assert(out.size() >= in.size());
  
  //thrust::copy(in.begin(), in.end(), out.begin());  
  cudaMemcpy((void *)out.data(), (const void *)thrust::raw_pointer_cast(in.data()), in.size() * sizeof(double), cudaMemcpyDeviceToHost);
  
  // No need to synchronize. For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed. See https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html.
  // cudaStreamSynchronize(0);
}

// void copy_vector(Eigen::VectorXd &out, const Eigen::VectorXd &in)
// {
//   out = in;
// }

void show_gpu_memory_usage(void)
{
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  std::cout << "free / total = "
	    << free << " B / " << total << " B ("
	    << free / (1024*1024) << " MB / " << total / (1024*1024) << " MB)\n";
}

cufftWrapperD2Z::cufftWrapperD2Z(int N_) : N(N_)
{
  cufftPlan3d(&plan, N_, N_, N_, CUFFT_D2Z);
  //std::cout << "plan initialized!\n";
}  

cufftWrapperD2Z::~cufftWrapperD2Z()
{
  cufftDestroy(plan);
  //std::cout << "plan destoyed!\n";
}

thrust::device_vector<double> cufftWrapperD2Z::execute(thrust::device_vector<double> &in)
{
  //std::cout << "executing plan!\n";
  thrust::device_vector<double> out(N * N * (N / 2 + 1) * 2);
  cufftExecD2Z(plan, thrust::raw_pointer_cast(in.data()), (cufftDoubleComplex *)thrust::raw_pointer_cast(out.data()));
  return out;
}


cufftWrapperBatchedD2Z::cufftWrapperBatchedD2Z(int N_) : N(N_)
{
  int rank = 3;
  int n[3] = {N_, N_, N_};
  int batch = 2;
  cufftPlanMany(&plan, rank, n, NULL, 0, 0, NULL, 0, 0, CUFFT_D2Z, batch);
  // std::cout << "plan initialized!\n";
  // size_t workSize;
  // cufftGetSize(plan, &workSize);
  // std::cout << "workSize = " << workSize << '\n';
}  

cufftWrapperBatchedD2Z::~cufftWrapperBatchedD2Z()
{
  cufftDestroy(plan);
  // std::cout << "plan destoyed!\n";
}

thrust::device_vector<double> cufftWrapperBatchedD2Z::execute(thrust::device_vector<double> &in)
{
  // std::cout << "executing plan!\n";
  thrust::device_vector<double> out(N * N * (N / 2 + 1) * 2 * 2);
  cufftExecD2Z(plan, thrust::raw_pointer_cast(in.data()), (cufftDoubleComplex *)thrust::raw_pointer_cast(out.data()));
  return out;
}


__device__
cufftDoubleComplex scale_callback(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
  int N = 384;
  int N3 = N * N * N;
  double r = ((cufftDoubleComplex *)dataIn)[offset].x / N3;
  double i = ((cufftDoubleComplex *)dataIn)[offset].y / N3;
  return make_cuDoubleComplex(r, i);
}

__device__ cufftCallbackLoadZ scale_callback_ptr = scale_callback;
cufftCallbackLoadZ hostCopyOfCallbackPtr;

cufftWrapper::cufftWrapper(int N_) : N(N_)
{
  cufftCreate(&plan_d2z);
  cufftCreate(&plan_batched_d2z);
  cufftCreate(&plan_z2d);

  cufftSetAutoAllocation(plan_d2z, 0);
  cufftSetAutoAllocation(plan_batched_d2z, 0);
  cufftSetAutoAllocation(plan_z2d, 0);

  size_t workSize_d2z, workSize_batched_d2z, workSize_z2d;

  cufftMakePlan3d(plan_d2z, N_, N_, N_, CUFFT_D2Z, &workSize_d2z);

  int rank = 3;
  int n[3] = {N_, N_, N_};
  int batch = 2;
  cufftMakePlanMany(plan_batched_d2z, rank, n, NULL, 0, 0, NULL, 0, 0, CUFFT_D2Z, batch, &workSize_batched_d2z);

  
  cufftMakePlan3d(plan_z2d, N_, N_, N_, CUFFT_Z2D, &workSize_z2d);
 
  size_t required_size = std::max({workSize_d2z, workSize_batched_d2z, workSize_z2d});
 
  work_area.resize(required_size / sizeof(double));
  
  //std::cout << "required_sizes = " << workSize_d2z << ", "
  //	    << workSize_batched_d2z << ", "
  //	    << workSize_z2d << '\n';
  //std::cout << "max required_size = " << required_size << '\n';
  //show_gpu_memory_usage();

  cufftSetWorkArea(plan_d2z, thrust::raw_pointer_cast(work_area.data()));
  cufftSetWorkArea(plan_batched_d2z, thrust::raw_pointer_cast(work_area.data()));
  cufftSetWorkArea(plan_z2d, thrust::raw_pointer_cast(work_area.data()));

  // cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, scale_callback_ptr, sizeof(hostCopyOfCallbackPtr));
  // cufftResult result = cufftXtSetCallback(plan_z2d, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX_DOUBLE, NULL);
  // std::cout << "no error = " << (result == CUFFT_SUCCESS) << '\n';
  // std::cout << "result = " << result << '\n';
}  

cufftWrapper::~cufftWrapper()
{
  cufftDestroy(plan_d2z);
  cufftDestroy(plan_batched_d2z);
  cufftDestroy(plan_z2d);
}

thrust::device_vector<double> cufftWrapper::execute_d2z(thrust::device_vector<double> &in)
{
  thrust::device_vector<double> out(N * N * (N / 2 + 1) * 2);
  cufftExecD2Z(plan_d2z, thrust::raw_pointer_cast(in.data()), (cufftDoubleComplex *)thrust::raw_pointer_cast(out.data()));
  return out;
}

thrust::device_vector<double> cufftWrapper::execute_batched_d2z(thrust::device_vector<double> &in)
{
  thrust::device_vector<double> out(N * N * (N / 2 + 1) * 2 * 2);
  cufftExecD2Z(plan_batched_d2z, thrust::raw_pointer_cast(in.data()), (cufftDoubleComplex *)thrust::raw_pointer_cast(out.data()));
  return out;
}

thrust::device_vector<double> cufftWrapper::execute_z2d(thrust::device_vector<double> &in)
{
  thrust::device_vector<double> out(N * N * N);
  cufftExecZ2D(plan_z2d, (cufftDoubleComplex *)thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()));
  return out;
}



cufftWrapperNoBatching::cufftWrapperNoBatching(int N_) : N(N_)
{
  cufftCreate(&plan_d2z);
  cufftCreate(&plan_z2d);

  cufftSetAutoAllocation(plan_d2z, 0);
  cufftSetAutoAllocation(plan_z2d, 0);

  size_t workSize_d2z, workSize_z2d;

  cufftMakePlan3d(plan_d2z, N_, N_, N_, CUFFT_D2Z, &workSize_d2z);

  cufftMakePlan3d(plan_z2d, N_, N_, N_, CUFFT_Z2D, &workSize_z2d);
 
  size_t required_size = std::max({workSize_d2z, workSize_z2d});
 
  work_area.resize(required_size / sizeof(double));
  
  cufftSetWorkArea(plan_d2z, thrust::raw_pointer_cast(work_area.data()));
  cufftSetWorkArea(plan_z2d, thrust::raw_pointer_cast(work_area.data()));
}  

cufftWrapperNoBatching::~cufftWrapperNoBatching()
{
  cufftDestroy(plan_d2z);
  cufftDestroy(plan_z2d);
}

thrust::device_vector<double> cufftWrapperNoBatching::execute_d2z(thrust::device_vector<double> &in)
{
  thrust::device_vector<double> out(N * N * (N / 2 + 1) * 2);
  cufftExecD2Z(plan_d2z, thrust::raw_pointer_cast(in.data()), (cufftDoubleComplex *)thrust::raw_pointer_cast(out.data()));
  return out;
}

thrust::device_vector<double> cufftWrapperNoBatching::execute_batched_d2z(thrust::device_vector<double> &in)
{
  thrust::device_vector<double> out(N * N * (N / 2 + 1) * 2 * 2);
  cufftExecD2Z(plan_d2z, thrust::raw_pointer_cast(in.data()), (cufftDoubleComplex *)thrust::raw_pointer_cast(out.data()));
  cufftExecD2Z(plan_d2z, thrust::raw_pointer_cast(in.data()) + N*N*N, ((cufftDoubleComplex *)thrust::raw_pointer_cast(out.data())) + N*N*(N/2+1));
  return out;
}

thrust::device_vector<double> cufftWrapperNoBatching::execute_z2d(thrust::device_vector<double> &in)
{
  thrust::device_vector<double> out(N * N * N);
  cufftExecZ2D(plan_z2d, (cufftDoubleComplex *)thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()));
  return out;
}

void cufftWrapperNoBatching::execute_inplace_z2d(thrust::device_vector<double> &inout)
{
  cufftExecZ2D(plan_z2d, (cufftDoubleComplex *)thrust::raw_pointer_cast(inout.data()), thrust::raw_pointer_cast(inout.data()));
}

/*
// Simple transformation kernel
__global__ void transformKernel(float* output,
                                cudaTextureObject_t texObj,
                                int width, int height,
                                float theta)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    // Transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // Read from texture and write to global memory
    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}
*/

 // Another transformation kernel
__global__ void transformKernel(float* output,
                                cudaTextureObject_t texObj)
{
  // Calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  float nx = (x+1) / 16.0;
  
  // Read from texture and write to global memory
  //output[x] = tex1Dfetch<float>(texObj, x);
  output[x] = tex1D<float>(texObj, nx);
  //output[x] = x;
}



void test_texture(void)
{
  int N = 8;
  // size_t size = N * sizeof(float);
  // float *h_data = (float *)std::malloc(size);
  // for (int i = 0; i < N; ++i) {
  //   h_data[i] = std::cos(i * 2 * std::numbers::pi / N);
  // }
  // float *dData = NULL;
  // checkCudaErrors(cudaMalloc((void **) &dData, size));

  Eigen::VectorXf vec_h(N);
  thrust::device_vector<float> vec_d(N);
  
  for(int i = 0; i < N; ++i) {
    vec_h[i] = std::cos(i * 2 * std::numbers::pi / N);
  }
  thrust::copy(vec_h.begin(), vec_h.end(), vec_d.begin());
  

  cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  
  cudaTextureObject_t texture_obj = 0;
  cudaResourceDesc resource_desc;
  cudaTextureDesc texture_desc;
  //cudaResourceViewDesc resource_view_desc;
  
  // resource_desc.resType = cudaResourceTypeLinear;
  // resource_desc.res.linear.devPtr = (void *)thrust::raw_pointer_cast(vec_d.data());
  // resource_desc.res.linear.desc = channelDesc;
  // resource_desc.res.linear.sizeInBytes = vec_d.size() * sizeof(float);
  cudaArray_t cuArray;
  //cudaMallocArray(&cuArray, &channelDesc, N);
  cudaMallocArray(&cuArray, &channelDesc, N);
  cudaMemcpyToArray(cuArray, 0, 0, vec_h.data(),
		    sizeof(float) * N, cudaMemcpyHostToDevice);
  

  // Specify texture
  memset(&resource_desc, 0, sizeof(resource_desc));
  resource_desc.resType = cudaResourceTypeArray;
  resource_desc.res.array.array = cuArray;


  // Specify texture object parameters
  memset(&texture_desc, 0, sizeof(texture_desc));
  texture_desc.addressMode[0] = cudaAddressModeWrap;
  //texture_desc.addressMode[1] = cudaAddressModeWrap;
  texture_desc.filterMode = cudaFilterModeLinear;
  texture_desc.readMode = cudaReadModeElementType;
  texture_desc.normalizedCoords = 1;

  //cudaBindTextureToArray(&texure_desc, cuArray);
  
  // Create texture object
  cudaCreateTextureObject(&texture_obj, &resource_desc, &texture_desc, NULL);

  
  Eigen::VectorXf out_h(2*N);
  thrust::device_vector<float> out_d(2*N);

  
  // Invoke kernel
  dim3 threadsperBlock(N);
  dim3 numBlocks(2);
  transformKernel<<<numBlocks, threadsperBlock>>>(thrust::raw_pointer_cast(out_d.data()), texture_obj);
  
  
  // Copy data from device back to host
  //thrust::copy(out_d.begin(), out_d.end(), out_h.begin());  
  cudaMemcpy((void *)out_h.data(), (const void *)thrust::raw_pointer_cast(out_d.data()), out_d.size() * sizeof(float), cudaMemcpyDeviceToHost);
  
  
  // Destroy texture object
  cudaDestroyTextureObject(texture_obj);

  std::cout << "in = " << vec_h.transpose() << '\n';
  std::cout << "out = " << out_h.transpose() << '\n';

  
  /*
  // Allocate array and copy image data
  cudaChannelFormatDesc channelDesc =
  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  checkCudaErrors(cudaMallocArray(&cuArray,
				  &channelDesc,
				  width,
				  height));
  checkCudaErrors(cudaMemcpyToArray(cuArray,
  0,
  0,
  hData,
  size,
  cudaMemcpyHostToDevice));

  // Bind the array to the texture
  //checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

  */

}

