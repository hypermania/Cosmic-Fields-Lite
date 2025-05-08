#include "proca.hpp"

Eigen::VectorXd generate_gaussian_random_proca_field(const long long int N, const double L, const Spectrum &P)
{
  const long long int lattice_size = N*N*N;
  Eigen::VectorXd field(3 * lattice_size);
  field.segment(0, lattice_size) = generate_gaussian_random_field(N, L, P);
  field.segment(lattice_size, lattice_size) = generate_gaussian_random_field(N, L, P);
  field.segment(2 * lattice_size, lattice_size) = generate_gaussian_random_field(N, L, P);
  return field;
}

void proca_project_to_transverse(const long long int N, Eigen::VectorXd &fields, fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper)
{
  const long long int lattice_size = N*N*N;

  Eigen::VectorXd Ax_k;
  Eigen::VectorXd Ay_k;
  Eigen::VectorXd Az_k;
  
  {
    Eigen::VectorXd Ai = fields.segment(0, lattice_size);
    Ax_k = fft_wrapper.execute_d2z(Ai);
    Ai = fields.segment(lattice_size, lattice_size);
    Ay_k = fft_wrapper.execute_d2z(Ai);
    Ai = fields.segment(2 * lattice_size, lattice_size);
    Az_k = fft_wrapper.execute_d2z(Ai);
  }

  // M_ij = I_ij - k_i k_j / k^2
  // new_Ax_k = M_xx Ax_k + M_xy Ay_k + M_xz Az_k

  Eigen::VectorXd M_x_k(Ax_k.size() / 2);
  Eigen::VectorXd M_y_k(Ax_k.size() / 2);
  Eigen::VectorXd M_z_k(Ax_k.size() / 2);

  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c <= N/2; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;

	if(s_sqr == 0) {
	  M_x_k(idx) = 1.0;
	  M_y_k(idx) = 0.0;
	  M_z_k(idx) = 0.0;
	  continue;
	}
	
	double k_a = (a<=N/2) ? a : (a-N);
	double k_b = (b<=N/2) ? b : (b-N);
	double k_c = c;
	M_x_k(idx) = 1.0 - k_a * k_a / s_sqr;
	M_y_k(idx) = - k_a * k_b / s_sqr;
	M_z_k(idx) = - k_a * k_c / s_sqr;
	
	// double f_k_re = f_k(2 * idx + 0);
	// double f_k_im = f_k(2 * idx + 1);
	// spectrum(s_sqr) += f_k_re * f_k_re + f_k_im * f_k_im;
      }
    }
  }

}
