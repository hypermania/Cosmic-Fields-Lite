#include "random_field.hpp"


#include <cassert>
#include <numbers>
#include <fftw3.h>
//#include <complex>
//#include <iostream>
//#include <cstdlib>

//#include "fdm3d.hpp"

namespace RandomNormal
{
  static std::mt19937 mt{ get_generator_from_device() };
  static std::normal_distribution<double> dist{0.0, 1.0};
  void set_generator_seed(std::mt19937::result_type seed)// long unsigned int seed)
  {
    mt = std::mt19937{ seed };
  }

  std::mt19937 get_generator_from_device()
  {
    std::random_device rd{};
    return std::mt19937{ rd() };
  }

  double generate_random_normal()
  {
    return dist(mt);
  }
}


// Generate inhomogeneous gaussian random field, where at point x the standard deviation of phi is modulated by exp(Psi).
Eigen::VectorXd generate_inhomogeneous_gaussian_random_field(const long long int N, const double L, const Eigen::VectorXd &Psi, const Spectrum &P) {
  assert((N > 0 && N % 2 == 0) && "N is not even.");

  using namespace std::numbers;
  using namespace Eigen;

  Eigen::VectorXd phi(N*N*N);
  // 1.1 Generate realization of a unit amplitude white noise field
  for(long long int i = 0; i < phi.size(); ++i){
    phi(i) = (1.0/(double)(N*N*N)) * RandomNormal::generate_random_normal();
  }
  
  // 1.2 Modulate by Psi
  phi.array() = phi.array() * Psi.array().exp();

  // 2. Perform Fourier transform
  fftw_complex *phi_k = (fftw_complex *)fftw_malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
  
  fftw_plan plan = fftw_plan_dft_r2c_3d(N, N, N, phi.data(), phi_k, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);


  // 3. Multiply by spectral data
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c <= N / 2; ++c){
	long long int a_shifted = (a<=N/2) ? a : (a-N);
	long long int b_shifted = (b<=N/2) ? b : (b-N);
	long long int c_shifted = (c<=N/2) ? c : (c-N);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	double abs_k = sqrt(s_sqr) * 2 * pi / L;
	double sqrt_P_k = sqrt(P(abs_k));
	phi_k[N*(N/2+1)*a + (N/2+1)*b + c][0] *= sqrt_P_k;
	phi_k[N*(N/2+1)*a + (N/2+1)*b + c][1] *= sqrt_P_k;
      }
    }
  }

  // 4. Perform inverse Fourier transform
  plan = fftw_plan_dft_c2r_3d(N, N, N, phi_k, phi.data(), FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  fftw_free(phi_k);

  return phi;
}


Eigen::VectorXd generate_gaussian_random_field(const long long int N, const double L, const Spectrum &P) {
  using namespace Eigen;
  const VectorXd Psi = VectorXd::Constant(N*N*N, 0);
  return generate_inhomogeneous_gaussian_random_field(N, L, Psi, P);
}


Spectrum power_law_with_cutoff_given_amplitude_3d(const long long int N, const double L, const double sigma, const double k_ast, const double alpha) {
  using namespace std::numbers;
  assert((alpha >= -3.0) && "Error: IR divergent choice of alpha.");
  assert((k_ast * L / (2 * pi) < 1.0) && "Error: k_ast is less than k_IR (the smallest wavenumber given box size).");
  double P_k_ast = alpha == -3.0
    ? 2 * pi * pi * sigma * sigma / (k_ast * k_ast * k_ast * log(k_ast * L / (2 * pi)))
    : (3 + alpha) * 2 * pi * pi * sigma * sigma * (L * L * L * pow(k_ast * L, alpha)) / (pow(k_ast * L, alpha + 3) - pow(2 * pi, alpha + 3));
  P_k_ast *= (N / L) * (N / L) * (N / L);
  return [=](double k){ return k == 0.0 ? 0.0 : (P_k_ast * (k <= k_ast ? pow(k/k_ast, alpha) : 0.0)); };
}

Spectrum broken_power_law_given_amplitude_3d(const long long int N, const double L, const double sigma, const double k_ast, const double alpha, const double beta) {
  using namespace std::numbers;
  assert((alpha >= -3.0) && "Error: IR divergent choice of alpha.");
  assert((k_ast * L / (2 * pi) < 1.0) && "Error: k_ast is less than k_IR (the smallest wavenumber given box size).");
  double P_k_ast = alpha == -3.0
    ? (-2 * pi * pi) * (3 + beta) * sigma * sigma / (pow(k_ast, 3) * (1 - (3 + beta) * log(k_ast * L / (2 * pi))))
    : (-2 * pi * pi) * pow(L, 3) * pow(k_ast * L, alpha) * (3 + alpha) * (3 + beta) * sigma * sigma / (pow(k_ast * L, alpha + 3) * (alpha - beta) + pow(2 * pi, alpha + 3) * (3 + beta));
  P_k_ast *= (N / L) * (N / L) * (N / L);
  return [=](double k){ return k == 0.0 ? 0.0 : (P_k_ast * (k <= k_ast ? pow(k/k_ast, alpha) : pow(k/k_ast, beta))); };
}

Spectrum scale_invariant_spectrum_3d(const long long int N, const double L, const double A_s) {
  using namespace std::numbers;
  double k0 = 1.0;
  double P_k0 = A_s / ( (k0 * k0 * k0) / (2 * pi * pi) );
  P_k0 *= (N / L) * (N / L) * (N / L);
  return [=](double k){ return k == 0.0 ? 0.0 : (P_k0 * pow(k/k0, -3)); };
}

Spectrum test_spectrum(const long long int N, const double L, const double sigma, const double k_ast, const double alpha) {
  assert((alpha >= -3.0) && "Error: IR divergent choice of alpha.");
  using namespace std::numbers;
  return [=](double k){ return ((k <= k_ast)) ? 1.0 : 0.0; };
}

Spectrum to_deriv_spectrum(const double m, const Spectrum &P_f) {
  return [m, &P_f](const double k){ return (k*k + m*m) * P_f(k); };
}

Spectrum to_deriv_spectrum(const double m, const double a, const Spectrum &P_f) {
  return [m, a, &P_f](const double k){ return (k*k / (a*a) + m*m) * P_f(k); };
}
