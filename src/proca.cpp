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

void ProcaTransverseProjector::init(void)
{
  
  M_xx_k.resize(fourier_size);
  M_xy_k.resize(fourier_size);
  M_xz_k.resize(fourier_size);
  M_yy_k.resize(fourier_size);
  M_yz_k.resize(fourier_size);
  M_zz_k.resize(fourier_size);

  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c <= N/2; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;

	if(s_sqr == 0) {
	  M_xx_k.segment(2 * idx, 2).array() = 1.0;
	  M_xy_k.segment(2 * idx, 2).array() = 0.0;
	  M_xz_k.segment(2 * idx, 2).array() = 0.0;
	  M_yy_k.segment(2 * idx, 2).array() = 1.0;
	  M_yz_k.segment(2 * idx, 2).array() = 0.0;
	  M_zz_k.segment(2 * idx, 2).array() = 1.0;
	  continue;
	}
	
	double k_a = (a<=N/2) ? a : (a-N);
	double k_b = (b<=N/2) ? b : (b-N);
	double k_c = c;
	M_xx_k.segment(2 * idx, 2).array() = 1.0 - k_a * k_a / s_sqr;
	M_xy_k.segment(2 * idx, 2).array() = - k_a * k_b / s_sqr;
	M_xz_k.segment(2 * idx, 2).array() = - k_a * k_c / s_sqr;
	M_yy_k.segment(2 * idx, 2).array() = 1.0 - k_b * k_b / s_sqr;
	M_yz_k.segment(2 * idx, 2).array() = - k_b * k_c / s_sqr;
	M_zz_k.segment(2 * idx, 2).array() = 1.0 - k_c * k_c / s_sqr;
      }
    }
  }
  
}


void ProcaTransverseProjector::proca_project_to_transverse(Eigen::VectorXd &fields, fftWrapperDispatcher<Eigen::VectorXd>::Generic &fft_wrapper)
{
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

  {
    Eigen::VectorXd new_Ax_k = (M_xx_k.array() * Ax_k.array() + M_xy_k.array() * Ay_k.array() + M_xz_k.array() * Az_k.array()).matrix();
    fields.segment(0, lattice_size) = fft_wrapper.execute_z2d(new_Ax_k) / lattice_size;
  }

  {
    Eigen::VectorXd new_Ay_k = (M_xy_k.array() * Ax_k.array() + M_yy_k.array() * Ay_k.array() + M_yz_k.array() * Az_k.array()).matrix();
    fields.segment(lattice_size, lattice_size) = fft_wrapper.execute_z2d(new_Ay_k) / lattice_size;
  }

  {
    Eigen::VectorXd new_Az_k = (M_xz_k.array() * Ax_k.array() + M_yz_k.array() * Ay_k.array() + M_zz_k.array() * Az_k.array()).matrix();
    fields.segment(2 * lattice_size, lattice_size) = fft_wrapper.execute_z2d(new_Az_k) / lattice_size;
  }
}


// TODO
ProcaEquation::Vector ProcaEquation::compute_At(Workspace &workspace, const double t)
{
  using namespace Eigen;
  using namespace std::numbers;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  const long long int lattice_size = N*N*N;
  const long long int fourier_size = 2*N*N*(N/2+1);
  
  VectorXd At_k(fourier_size);

  // auto Ax = workspace.state.segment(0, lattice_size);
  // auto Ay = workspace.state.segment(lattice_size, lattice_size);
  // auto Az = workspace.state.segment(2 * lattice_size, lattice_size);
  auto dt_Ax = workspace.state.segment(3 * lattice_size, lattice_size);
  auto dt_Ay = workspace.state.segment(4 * lattice_size, lattice_size);
  auto dt_Az = workspace.state.segment(5 * lattice_size, lattice_size);
  
  VectorXd dt_Ax_k = workspace.fft_wrapper.execute_d2z(dt_Ax);
  VectorXd dt_Ay_k = workspace.fft_wrapper.execute_d2z(dt_Ay);
  VectorXd dt_Az_k = workspace.fft_wrapper.execute_d2z(dt_Az);

  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c <= N/2; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;
	
	double k_a = ((a<=N/2) ? a : (a-N)) * (2 * pi / L);
	double k_b = ((b<=N/2) ? b : (b-N)) * (2 * pi / L);
	double k_c = c * (2 * pi / L);
	double k = sqrt(static_cast<double>(s_sqr)) * (2 * pi / L);

	At_k(2 * idx + 0) = -(k_a * dt_Ax_k(2 * idx + 1) + k_b * dt_Ay_k(2 * idx + 1) + k_c * dt_Az_k(2 * idx + 1)) / (k*k + m*m) / (N*N*N);
	At_k(2 * idx + 1) = (k_a * dt_Ax_k(2 * idx + 0) + k_b * dt_Ay_k(2 * idx + 0) + k_c * dt_Az_k(2 * idx + 0)) / (k*k + m*m) / (N*N*N);
	
      }
    }
  }
  
  VectorXd At = workspace.fft_wrapper.execute_z2d(At_k);
  
  return At;
}


// TODO
ProcaEquation::Vector ProcaEquation::compute_energy_density(const Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));

  const long long int lattice_size = N*N*N;
  VectorXd rho(lattice_size);
  
  // VectorXd At(lattice_size);
  // auto &Ax = workspace.state.segment(0, lattice_size);
  // auto &Ay = workspace.state.segment(lattice_size, lattice_size);
  // auto &Az = workspace.state.segment(2 * lattice_size, lattice_size);
  // auto &dt_Ax = workspace.state.segment(3 * lattice_size, lattice_size);
  // auto &dt_Ay = workspace.state.segment(4 * lattice_size, lattice_size);
  // auto &dt_Az = workspace.state.segment(5 * lattice_size, lattice_size);

    
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      rho(seqN(IDX_OF(N, a, b, 0), N)) = 0.5 *
	( workspace.state(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).cwiseAbs2()
	  + m * m * workspace.state(seqN(IDX_OF(N, a, b, 0), N)).cwiseAbs2()
	  + 0.25 * inv_h_sqr *
	  ( (workspace.state(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
	     - workspace.state(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))).cwiseAbs2()
	    + (workspace.state(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
	       - workspace.state(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N))).cwiseAbs2() )
	  );
      rho(seqN(IDX_OF(N, a, b, 1), N-2)) += 0.5 * 0.25 * inv_h_sqr *
	(workspace.state(seqN(IDX_OF(N, a, b, 2), N-2))
	 - workspace.state(seqN(IDX_OF(N, a, b, 0), N-2))).cwiseAbs2();
      rho(IDX_OF(N, a, b, 0)) += 0.5 * 0.25 * inv_h_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 1)) - workspace.state(IDX_OF(N, a, b, N-1)), 2);
      rho(IDX_OF(N, a, b, N-1)) += 0.5 * 0.25 * inv_h_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 0)) - workspace.state(IDX_OF(N, a, b, N-2)), 2);
    }
  }
  return rho;
}


// TODO
ProcaEquation::Vector ProcaEquation::compute_momentum_density(const Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  //  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  const double inv_two_h = 1.0 / (2.0 * L / N);
  const long long int field_size = N * N * N;
  
  VectorXd q(3 * field_size);
  auto &varphi = workspace.state.head(field_size);
  auto &dt_varphi = workspace.state.tail(field_size);
  // auto &q_x = q.head(field_size);
  // auto &q_y = q.segment(field_size, field_size);
  // auto &q_z = q.tail(field_size);
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      q(seqN(IDX_OF(N, a, b, 0), N)).array() = - dt_varphi(seqN(IDX_OF(N, a, b, 0), N)).array() * inv_two_h
	* ( varphi(seqN(IDX_OF(N, (a+1)%N, b, 0), N)) - varphi(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N)) ).array();

      q(seqN(field_size + IDX_OF(N, a, b, 0), N)).array() = - dt_varphi(seqN(IDX_OF(N, a, b, 0), N)).array() * inv_two_h
	* ( varphi(seqN(IDX_OF(N, a, (b+1)%N, 0), N)) - varphi(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N)) ).array();

      q(seqN(2*field_size + IDX_OF(N, a, b, 1), N-2)).array() = - dt_varphi(seqN(IDX_OF(N, a, b, 1), N-2)).array() * inv_two_h
	* ( varphi(seqN(IDX_OF(N, a, b, 2), N-2)) - varphi(seqN(IDX_OF(N, a, b, 0), N-2)) ).array();

      q(2*field_size + IDX_OF(N, a, b, 0)) = - dt_varphi(IDX_OF(N, a, b, 0)) * inv_two_h
	* ( varphi(IDX_OF(N, a, b, 1)) - varphi(IDX_OF(N, a, b, N-1)) );

      q(2*field_size + IDX_OF(N, a, b, N-1)) = - dt_varphi(IDX_OF(N, a, b, N-1)) * inv_two_h
	* ( varphi(IDX_OF(N, a, b, 0)) - varphi(IDX_OF(N, a, b, N-2)) );
      
    }
  }
  return q;
}
