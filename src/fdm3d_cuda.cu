#include "fdm3d_cuda.cuh"

__global__
void sum_power_kernel(const double *fft, double *spectrum, const long long int N)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  int half_N = N/2;
  int a_shifted = (a<half_N) ? a : (N-a);
  int b_shifted = (b<half_N) ? b : (N-b);
  int c_shifted = (c<half_N) ? c : (N-c);
  
  int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c_shifted);
  
  double f_k_re = fft[offset_k];
  double f_k_im = fft[offset_k + 1];
  
  int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c_shifted * c_shifted;
  
  double val = f_k_re * f_k_re + f_k_im * f_k_im;
  
  // This code will not produce the same result each time because fp addition is not associative.
  atomicAdd(&spectrum[s_sqr], val);
}



__global__
void sum_mode_power_kernel(const double *ffts, double *spectrum,
			   const long long int N, const double L, const double m, const double a_t)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  int half_N = N/2;
  int a_shifted = (a<half_N) ? a : (N-a);
  int b_shifted = (b<half_N) ? b : (N-b);
  int c_shifted = (c<half_N) ? c : (N-c);
  
  int fft_size = 2 * N * N * (half_N + 1);
  int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c_shifted);
  
  double f_k_re = ffts[offset_k];
  double f_k_im = ffts[offset_k + 1];

  double dtf_k_re = ffts[fft_size + offset_k];
  double dtf_k_im = ffts[fft_size + offset_k + 1];

  int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c_shifted * c_shifted;
  // double omega_k_sqr = m * m + (2 * pi / L) * (2 * pi / L) * s_sqr;
  
  double val = (f_k_re * f_k_re + f_k_im * f_k_im)
    + (dtf_k_re * dtf_k_re + dtf_k_im * dtf_k_im) / (m * m + (2 * pi / L) * (2 * pi / L) * s_sqr / (a_t * a_t));
  
  // This code will not produce the same result each time because fp addition is not associative.
  atomicAdd(&spectrum[s_sqr], val);
}


__global__
void compute_laplacian_kernel(const double *in, double *out, const long long int N, const double inv_h_sqr)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;

  int idx1 = IDX_OF(N, a, b, c);
  out[idx1] = inv_h_sqr * (-6.0 * in[idx1]
			   + in[IDX_OF(N, (a+1)%N, b, c)]
			   + in[IDX_OF(N, (a+N-1)%N, b, c)]
			   + in[IDX_OF(N, a, (b+1)%N, c)]
			   + in[IDX_OF(N, a, (b+N-1)%N, c)]
			   + in[IDX_OF(N, a, b, (c+1)%N)]
			   + in[IDX_OF(N, a, b, (c+N-1)%N)]);
}


__global__
void compute_inverse_laplacian_kernel(double *inout, const long long int N, const double L)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  int half_N = N/2;
  if(c <= N/2) {
    int a_shifted = (a<half_N) ? a : (N-a);
    int b_shifted = (b<half_N) ? b : (N-b);
    //int c_shifted = (c<half_N) ? c : (N-c);
  
    //int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c_shifted);
    int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c);

    //int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c_shifted * c_shifted;
    int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c * c;

    double scale = (2 * pi / L) * (2 * pi / L) * (N*N*N);

    if(s_sqr == 0) {
      inout[offset_k] = 0;
      inout[offset_k + 1] = 0;
    } else {
      double k_sqr = s_sqr * scale;
      inout[offset_k] /= -k_sqr;
      inout[offset_k + 1] /= -k_sqr;
    }
  }
}


__global__
void cutoff_fourier_kernel(const double *in, double *out, const long long int N, const long long int M)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  double scale_diff = N / M;
  double scaling = 1.0 / (scale_diff * scale_diff * scale_diff);
  
  int half_M = M/2;
  int half_N = N/2;
  if(c <= M/2) {
    int a_N = (a<half_M) ? a : (N-(M-a));
    int b_N = (b<half_M) ? b : (N-(M-b));
    int c_N = c;
    
    int offset_k_M = 2 * (M*(half_M+1)*a + (half_M+1)*b + c);
    int offset_k_N = 2 * (N*(half_N+1)*a_N + (half_N+1)*b_N + c_N);

    out[offset_k_M] = in[offset_k_N] * scaling;
    out[offset_k_M + 1] = in[offset_k_N + 1] * scaling;
  }
}


thrust::device_vector<double> compute_mode_power_spectrum(const long long int N, const double L, const double m, const double a_t,
							  thrust::device_vector<double> &state,
							  fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper)
{
  assert(N % 8 == 0);
  
  thrust::device_vector<double> spectrum(3*(N/2)*(N/2)+1);

  thrust::device_vector<double> ffts = fft_wrapper.execute_batched_d2z(state);

  dim3 threadsPerBlock(8, 4, 4);
  dim3 numBlocks((int)N/8, (int)N/4, (int)N/4);
  sum_mode_power_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(ffts.data()), thrust::raw_pointer_cast(spectrum.data()), N, L, m, a_t);

  return spectrum;
}

thrust::device_vector<double> compute_power_spectrum(const long long int N,
						     thrust::device_vector<double> &f,
						     fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper)
{
  assert(N % 8 == 0);
  
  thrust::device_vector<double> spectrum(3*(N/2)*(N/2)+1);
  //auto fft = fft_d2z.execute(f);
  thrust::device_vector<double> fft = fft_wrapper.execute_d2z(f);

  dim3 threadsPerBlock(8, 4, 4);
  dim3 numBlocks((int)N/8, (int)N/4, (int)N/4);
  sum_power_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(fft.data()), thrust::raw_pointer_cast(spectrum.data()), N);
  
  return spectrum;
}

thrust::device_vector<double> compute_laplacian(const long long int N, const double L,
						thrust::device_vector<double> &f)
{
  thrust::device_vector<double> laplacian(f.size());
  const double inv_h_sqr = 1 / ((L / N) * (L / N));

  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  compute_laplacian_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(f.data()), thrust::raw_pointer_cast(laplacian.data()), N, inv_h_sqr);

  return laplacian;
}

thrust::device_vector<double> compute_inverse_laplacian(const long long int N, const double L,
							thrust::device_vector<double> &f,
							fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper)
{
  assert(N % 8 == 0);
  
  thrust::device_vector<double> fft = fft_wrapper.execute_d2z(f);

  dim3 threadsPerBlock(8, 4, 4);
  dim3 numBlocks((int)N/8, (int)N/4, (int)N/4);
  compute_inverse_laplacian_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(fft.data()), N, L);
  
  return fft_wrapper.execute_z2d(fft);
}

thrust::device_vector<double> compute_cutoff_fouriers(const long long int N, const long long int M,
						      const thrust::device_vector<double> &fft)
{
  assert((N % 8 == 0) && (M % 8 == 0));
  assert(N >= M && "The new grid must be coarser than the old grid.");
  
  thrust::device_vector<double> cutoff_fft(2 * M * M * (M / 2 + 1));
  
  dim3 threadsPerBlock_fft(8, 4, 4);
  dim3 numBlocks_fft((int)M/8, (int)M/4, (int)M/4/2+1);
  cutoff_fourier_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(fft.data()), thrust::raw_pointer_cast(cutoff_fft.data()), N, M);

  return cutoff_fft;
}

void compute_inverse_laplacian_test(const long long int N, const double L,
				    thrust::device_vector<double> &fft)
{
  assert(N % 8 == 0);

  dim3 threadsPerBlock(8, 4, 4);
  dim3 numBlocks((int)N/8, (int)N/4, (int)N/4);
  compute_inverse_laplacian_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(fft.data()), N, L);
}


