#include "wkb.hpp"

#include <cmath>

WKBSolutionForKleinGordonEquationInFRW::WKBSolutionForKleinGordonEquationInFRW(Workspace &workspace_, const double t_i_) : workspace(workspace_), t_i(t_i_)
{
  assert((workspace.cosmology.p == 1.0) && "Only radiation dominated cosmology is supported for now.");
  assert(workspace.cosmology.t1 == 1.0 / (2 * workspace.cosmology.H1));
  
  const double a_t = workspace.cosmology.a(t_i);
  const double H_t = workspace.cosmology.H(t_i);
  const int fft_size = 2 * workspace.N * workspace.N * (workspace.N / 2 + 1);
  phi_ffts = workspace.fft_wrapper.execute_batched_d2z(workspace.state);
  
  assert(phi_ffts.size() == 2 * fft_size);
    
  phi_ffts.head(fft_size) *= pow(a_t, 1.5);
  phi_ffts.tail(fft_size) = pow(a_t, 1.5) * phi_ffts.tail(fft_size) + 1.5 * H_t * phi_ffts.head(fft_size);



}


WKBSolutionForKleinGordonEquationInFRW::Vector WKBSolutionForKleinGordonEquationInFRW::evaluate_at(const double t)
{
  using namespace std::numbers;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const long long int fft_size = phi_ffts.size() / 2;
  //const long long int num_modes = fft_size / 2;
  //const int field_size = workspace.N * workspace.N * workspace.N;

  const double a1 = workspace.cosmology.a1;
  const double H1 = workspace.cosmology.H1;
  const double m = workspace.m;
  //const double t_i = this->t_i;
  const double t_f = t;
  
  
  auto compute_phase_integral =
    [=, this](const double k_sqr){
      return std::sqrt(H1 *
		       (8 * k_sqr * t_f + std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_f, 2)))) /
	(4. * a1 * H1) -
	std::sqrt(H1 *
		  (8 * k_sqr * t_i + std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_i, 2)))) /
	(4. * a1 * H1) +
	(std::sqrt(3) *
	 std::log((1 - (3 * std::pow(a1, 2) * H1 + 4 * k_sqr * t_f) /
		   (std::sqrt(3) * a1 *
		    std::sqrt(H1 * (8 * k_sqr * t_f +
				    std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_f, 2)))))) /
		  (1 - (3 * std::pow(a1, 2) * H1 + 4 * k_sqr * t_i) /
		   (std::sqrt(3) * a1 *
		    std::sqrt(H1 * (8 * k_sqr * t_i +
				    std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_i, 2)))))))) /
	8. -
	(std::sqrt(3) *
	 std::log((1 + (3 * std::pow(a1, 2) * H1 + 4 * k_sqr * t_f) /
		   (std::sqrt(3) * a1 *
		    std::sqrt(H1 * (8 * k_sqr * t_f +
				    std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_f, 2)))))) /
		  (1 + (3 * std::pow(a1, 2) * H1 + 4 * k_sqr * t_i) /
		   (std::sqrt(3) * a1 *
		    std::sqrt(H1 * (8 * k_sqr * t_i +
				    std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_i, 2)))))))) /
	8. +
	(k_sqr *
	 std::log((k_sqr +
		   a1 * m *
                   (4 * a1 * H1 * m * t_f +
                    std::sqrt(H1 * (8 * k_sqr * t_f +
				    std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_f, 2)))))) /
		  (k_sqr +
		   a1 * m *
                   (4 * a1 * H1 * m * t_i +
                    std::sqrt(H1 * (8 * k_sqr * t_i +
				    std::pow(a1, 2) * H1 * (3 + 16 * std::pow(m, 2) * std::pow(t_i, 2)))))))) /
	(4. * std::pow(a1, 2) * H1 * m);
    };
  
  //Eigen::VectorXd phase(num_modes);
  
  Eigen::VectorXd new_phi_fft(fft_size);
  Eigen::VectorXd new_dot_phi_fft(fft_size);
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c < N; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	double k_sqr = s_sqr * std::pow(2 * pi / L, 2);
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;
	double phase = compute_phase_integral(k_sqr);
	double omega_eff_i = std::sqrt(m * m + k_sqr / (2 * a1 * a1 * H1 * t_i) + 3 / (16 * t_i * t_i));
	//double omega_eff_f = std::sqrt(m * m + k_sqr / (2 * a1 * a1 * H1 * t_f) + 3 / (16 * t_f * t_f));
	// Ignoring the sqrt(omega) factor in front for now
	double cos_val = cos(phase);
	double sin_val = sin(phase);
	new_phi_fft(2 * idx) = phi_ffts(2 * idx) * cos_val + phi_ffts(fft_size + 2 * idx) * sin_val / std::sqrt(omega_eff_i);
	new_phi_fft(2 * idx + 1) = phi_ffts(2 * idx + 1) * cos_val + phi_ffts(fft_size + 2 * idx + 1) * sin_val / std::sqrt(omega_eff_i);
	new_dot_phi_fft(2 * idx) = phi_ffts(fft_size + 2 * idx) * cos_val - phi_ffts(2 * idx) * sin_val * std::sqrt(omega_eff_i);
	new_dot_phi_fft(2 * idx + 1) = phi_ffts(fft_size + 2 * idx + 1) * cos_val - phi_ffts(2 * idx + 1) * sin_val * std::sqrt(omega_eff_i);
      }
    }
  }

  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  
  Eigen::VectorXd new_phi = workspace.fft_wrapper.execute_z2d(new_phi_fft) / (N*N*N);
  Eigen::VectorXd new_dot_phi = workspace.fft_wrapper.execute_z2d(new_dot_phi_fft) / (N*N*N);
  Eigen::VectorXd new_solution(2 * N*N*N);
  new_solution.head(N*N*N) = new_phi / pow(a_t, 1.5);
  new_solution.tail(N*N*N) = new_dot_phi / pow(a_t, 1.5) - 1.5 * H_t * new_phi / pow(a_t, 1.5);
  
  return new_solution;
}
