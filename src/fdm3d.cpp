#include "fdm3d.hpp"

#include <fftw3.h>
#include <numbers>


Eigen::VectorXd compute_power_spectrum(const long long int N,
				       Eigen::VectorXd &f,
				       fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper)
{
  Eigen::VectorXd spectrum(3*(N/2)*(N/2)+1);
  spectrum.array() = 0.0;

  // Perform Fourier transform
  auto f_k = fft_wrapper.execute_d2z(f);
  
  // Summarize spectral data
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c < N; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;
	double f_k_re = f_k(2 * idx + 0);
	double f_k_im = f_k(2 * idx + 1);
	spectrum(s_sqr) += f_k_re * f_k_re + f_k_im * f_k_im;
      }
    }
  }

  return spectrum;
}

Eigen::VectorXd compute_mode_power_spectrum(const long long int N, const double L, const double m, const double a_t,
					    Eigen::VectorXd &state,
					    fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper)
{
  using namespace std::numbers;
  
  Eigen::VectorXd spectrum(3*(N/2)*(N/2)+1);
  spectrum.array() = 0.0;

  // Perform Fourier transform
  const int N_3 = state.size() / 2;
  Eigen::VectorXd phi = state.head(N_3);
  Eigen::VectorXd dt_phi = state.tail(N_3);
  auto phi_k = fft_wrapper.execute_d2z(phi);
  auto dt_phi_k = fft_wrapper.execute_d2z(dt_phi);
  
  // Summarize spectral data
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c < N; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;
	double phi_k_re = phi_k(2 * idx + 0);
	double phi_k_im = phi_k(2 * idx + 1);
	double dt_phi_k_re = dt_phi_k(2 * idx + 0);
	double dt_phi_k_im = dt_phi_k(2 * idx + 1);
	spectrum(s_sqr) += phi_k_re * phi_k_re + phi_k_im * phi_k_im
	  + (dt_phi_k_re * dt_phi_k_re + dt_phi_k_im * dt_phi_k_im) / (m * m + (2 * pi / L) * (2 * pi / L) * s_sqr / (a_t * a_t));
      }
    }
  }

  return spectrum;
}



Eigen::VectorXd compute_field_with_scaled_fourier_modes(const long long int N, const double L,
							Eigen::VectorXd &f,
							std::function<double(const double)> kernel,
							fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper)
{
  using namespace std::numbers;
  // Perform Fourier transform
  auto f_k = fft_wrapper.execute_d2z(f);
  
  // Scale Fourier modes
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c <= N/2; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;
	double scale = kernel(sqrt(static_cast<double>(s_sqr)) * (2 * pi / L)) / (N * N * N);
	f_k(2 * idx + 0) *= scale;
	f_k(2 * idx + 1) *= scale;
      }
    }
  }

  return fft_wrapper.execute_z2d(f_k);
}


Eigen::VectorXd compute_cutoff_fouriers(const long long int N, const long long int M,
					Eigen::VectorXd &fft)
{
  assert(N >= M && "The new grid must be coarser than the old grid.");
  
  const double scale_diff = static_cast<double>(N) / static_cast<double>(M);
  Eigen::VectorXd cutoff_fft(2 * M * M * (M / 2 + 1));
  
  for(int a = 0; a < M; ++a){
    for(int b = 0; b < M; ++b){
      for(int c = 0; c <= M/2; ++c){
	// This scheme populates modes at a,b,c=M/2 on the new grid.
	// However, the corresponding modes on the old grid can be a,b,c=M/2 or a,b,c=N-(M/2).
	int a_N = (a<M/2) ? a : (N-(M-a));
	int b_N = (b<M/2) ? b : (N-(M-b));
	int c_N = c;
	
	int idx_M = M*(M/2+1)*a + (M/2+1)*b + c;
	int idx_N = N*(N/2+1)*a_N + (N/2+1)*b_N + c_N;
	cutoff_fft(2 * idx_M) = fft(2 * idx_N) / pow(scale_diff, 3);
	cutoff_fft(2 * idx_M + 1) = fft(2 * idx_N + 1) / pow(scale_diff, 3);
      }
    }
  }
  return cutoff_fft;
}


Eigen::VectorXd compute_power_spectrum(const long long int N, Eigen::VectorXd &phi)
{
  Eigen::VectorXd spectrum(3*(N/2)*(N/2)+1);
  spectrum.array() = 0.0;

  // Perform Fourier transform
  fftw_complex *phi_k = (fftw_complex *)fftw_malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
  
  fftw_plan plan = fftw_plan_dft_r2c_3d(N, N, N, phi.data(), phi_k, FFTW_ESTIMATE);
  fftw_execute(plan);

  
  // Summarize spectral data
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c < N; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	double phi_k_val_0 = phi_k[N*(N/2+1)*a + (N/2+1)*b + c_shifted][0];
	double phi_k_val_1 = phi_k[N*(N/2+1)*a + (N/2+1)*b + c_shifted][1];
	spectrum(s_sqr) += phi_k_val_0*phi_k_val_0 + phi_k_val_1*phi_k_val_1;
      }
    }
  }

  //spectrum /= (double)(N*N*N);
  
  // Clean up
  fftw_destroy_plan(plan);  
  fftw_free(phi_k);

  return spectrum;

}


Eigen::VectorXd compute_fourier(const long long int N, const double L, Eigen::VectorXd &phi)
{
  // Perform Fourier transform
  fftw_complex *phi_k = (fftw_complex *)fftw_malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
  fftw_plan plan = fftw_plan_dft_r2c_3d(N, N, N, phi.data(), phi_k, FFTW_ESTIMATE);
  fftw_execute(plan);
  
  Eigen::VectorXd fourier(2 * N * N * (N / 2 + 1));
  memcpy((void *)fourier.data(), (void *)phi_k, N * N * (N / 2 + 1) * sizeof(fftw_complex));
  
  // Clean up
  fftw_destroy_plan(plan);  
  fftw_free(phi_k);

  return fourier;
}


Eigen::VectorXd compute_laplacian(const long long int N, const double L, const Eigen::VectorXd &f)
{
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  Eigen::VectorXd laplacian(f.size());
  for(int a = 0; a < N; ++a) {
    for(int b = 0; b < N; ++b) {
      for(int c = 0; c < N; ++c) {
	laplacian(IDX_OF(N, a, b, c)) =
	  inv_h_sqr * (-6.0 * f(IDX_OF(N, a, b, c))
		       + f(IDX_OF(N, (a+1)%N, b, c))
		       + f(IDX_OF(N, (a+N-1)%N, b, c))
		       + f(IDX_OF(N, a, (b+1)%N, c))
		       + f(IDX_OF(N, a, (b+N-1)%N, c))
		       + f(IDX_OF(N, a, b, (c+1)%N))
		       + f(IDX_OF(N, a, b, (c+N-1)%N)) );
      }
    }
  }

  return laplacian;
}

Eigen::VectorXd compute_inverse_laplacian(const long long int N, const double L,
					  Eigen::VectorXd &f,
					  fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper)
{
  using namespace std::numbers;
  auto fft = fft_wrapper.execute_d2z(f);

  // Compute inverse laplacian
  for(int a = 0; a < N; ++a) {
    for(int b = 0; b < N; ++b) {
      for(int c = 0; c < N; ++c) {
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
	    fft(offset_k) = 0;
	    fft(offset_k + 1) = 0;
	  } else {
	    double k_sqr = s_sqr * scale;
	    fft(offset_k) /= -k_sqr;
	    fft(offset_k + 1) /= -k_sqr;
	  }
	}
      }
    }
  }

  return fft_wrapper.execute_z2d(fft);
}
