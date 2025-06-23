#include "proca.hpp"
#include "io.hpp"

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


ProcaEquation::Vector ProcaEquation::compute_At(Workspace &workspace, const double t)
{
  using namespace Eigen;
  using namespace std::numbers;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  // const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
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

	// At_k = (- i k . dt_A_k) / (k^2 + m^2)
	At_k(2 * idx + 0) = (k_a * dt_Ax_k(2 * idx + 1) + k_b * dt_Ay_k(2 * idx + 1) + k_c * dt_Az_k(2 * idx + 1)) / (k*k + m*m) / (N*N*N);
	At_k(2 * idx + 1) = -(k_a * dt_Ax_k(2 * idx + 0) + k_b * dt_Ay_k(2 * idx + 0) + k_c * dt_Az_k(2 * idx + 0)) / (k*k + m*m) / (N*N*N);
	
      }
    }
  }
  
  VectorXd At = workspace.fft_wrapper.execute_z2d(At_k);
  
  return At;
}


ProcaEquation::Vector ProcaEquation::compute_energy_density(Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  // const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  const double h_inv = N / L;
  const long long int lattice_size = N*N*N;
  VectorXd rho(lattice_size);
  
  VectorXd At = compute_At(workspace, t);
  auto Ax = workspace.state.segment(0, lattice_size);
  auto Ay = workspace.state.segment(lattice_size, lattice_size);
  auto Az = workspace.state.segment(2 * lattice_size, lattice_size);
  auto dt_Ax = workspace.state.segment(3 * lattice_size, lattice_size);
  auto dt_Ay = workspace.state.segment(4 * lattice_size, lattice_size);
  auto dt_Az = workspace.state.segment(5 * lattice_size, lattice_size);

    
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      
      rho.segment(IDX_OF(N, a, b, 1), N-2) = ((0.5))*(((((0.5))*((h_inv)*((Ax.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2))+((((-1))*(Ax.segment(IDX_OF(N,a,(b+1)%N,1),N-2)))+((((-1))*(Ay.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2)))+(Ay.segment(IDX_OF(N,(a+1)%N,b,1),N-2))))))).cwiseAbs2())+(((pow(m,(2)))*(((At.segment(IDX_OF(N,a,b,1),N-2)).cwiseAbs2())+(((Ax.segment(IDX_OF(N,a,b,1),N-2)).cwiseAbs2())+(((Ay.segment(IDX_OF(N,a,b,1),N-2)).cwiseAbs2())+((Az.segment(IDX_OF(N,a,b,1),N-2)).cwiseAbs2())))))+(((((0.5))*((h_inv)*((Ay.segment(IDX_OF(N,a,b,0),N-2))+((((-1))*(Ay.segment(IDX_OF(N,a,b,2),N-2)))+((((-1))*(Az.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2)))+(Az.segment(IDX_OF(N,a,(b+1)%N,1),N-2))))))).cwiseAbs2())+(((((0.5))*((h_inv)*((Ax.segment(IDX_OF(N,a,b,0),N-2))+((((-1))*(Ax.segment(IDX_OF(N,a,b,2),N-2)))+((((-1))*(Az.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2)))+(Az.segment(IDX_OF(N,(a+1)%N,b,1),N-2))))))).cwiseAbs2())+((((((0.5))*((h_inv)*((At.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2))+(((-1))*(At.segment(IDX_OF(N,(a+1)%N,b,1),N-2))))))+(dt_Ax.segment(IDX_OF(N,a,b,1),N-2))).cwiseAbs2())+((((((0.5))*((h_inv)*((At.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2))+(((-1))*(At.segment(IDX_OF(N,a,(b+1)%N,1),N-2))))))+(dt_Ay.segment(IDX_OF(N,a,b,1),N-2))).cwiseAbs2())+(((((0.5))*((h_inv)*((At.segment(IDX_OF(N,a,b,0),N-2))+(((-1))*(At.segment(IDX_OF(N,a,b,2),N-2))))))+(dt_Az.segment(IDX_OF(N,a,b,1),N-2))).cwiseAbs2())))))));

      rho(IDX_OF(N, a, b, 0)) = ((0.5))*((pow(((0.5))*((h_inv)*((Ax(IDX_OF(N,a,(b+N-1)%N,0)))+((((-1))*(Ax(IDX_OF(N,a,(b+1)%N,0))))+((((-1))*(Ay(IDX_OF(N,(a+N-1)%N,b,0))))+(Ay(IDX_OF(N,(a+1)%N,b,0))))))),2))+(((pow(m,(2)))*((pow(At(IDX_OF(N,a,b,0)),2))+((pow(Ax(IDX_OF(N,a,b,0)),2))+((pow(Ay(IDX_OF(N,a,b,0)),2))+(pow(Az(IDX_OF(N,a,b,0)),2))))))+((pow(((0.5))*((h_inv)*((Ay(IDX_OF(N,a,b,N-1)))+((((-1))*(Ay(IDX_OF(N,a,b,1))))+((((-1))*(Az(IDX_OF(N,a,(b+N-1)%N,0))))+(Az(IDX_OF(N,a,(b+1)%N,0))))))),2))+((pow(((0.5))*((h_inv)*((Ax(IDX_OF(N,a,b,N-1)))+((((-1))*(Ax(IDX_OF(N,a,b,1))))+((((-1))*(Az(IDX_OF(N,(a+N-1)%N,b,0))))+(Az(IDX_OF(N,(a+1)%N,b,0))))))),2))+((pow((((0.5))*((h_inv)*((At(IDX_OF(N,(a+N-1)%N,b,0)))+(((-1))*(At(IDX_OF(N,(a+1)%N,b,0)))))))+(dt_Ax(IDX_OF(N,a,b,0))),2))+((pow((((0.5))*((h_inv)*((At(IDX_OF(N,a,(b+N-1)%N,0)))+(((-1))*(At(IDX_OF(N,a,(b+1)%N,0)))))))+(dt_Ay(IDX_OF(N,a,b,0))),2))+(pow((((0.5))*((h_inv)*((At(IDX_OF(N,a,b,N-1)))+(((-1))*(At(IDX_OF(N,a,b,1)))))))+(dt_Az(IDX_OF(N,a,b,0))),2))))))));

      rho(IDX_OF(N, a, b, N-1)) = ((0.5))*((pow(((0.5))*((h_inv)*((Ax(IDX_OF(N,a,(b+N-1)%N,N-1)))+((((-1))*(Ax(IDX_OF(N,a,(b+1)%N,N-1))))+((((-1))*(Ay(IDX_OF(N,(a+N-1)%N,b,N-1))))+(Ay(IDX_OF(N,(a+1)%N,b,N-1))))))),2))+(((pow(m,(2)))*((pow(At(IDX_OF(N,a,b,N-1)),2))+((pow(Ax(IDX_OF(N,a,b,N-1)),2))+((pow(Ay(IDX_OF(N,a,b,N-1)),2))+(pow(Az(IDX_OF(N,a,b,N-1)),2))))))+((pow(((0.5))*((h_inv)*((Ay(IDX_OF(N,a,b,N-2)))+((((-1))*(Ay(IDX_OF(N,a,b,0))))+((((-1))*(Az(IDX_OF(N,a,(b+N-1)%N,N-1))))+(Az(IDX_OF(N,a,(b+1)%N,N-1))))))),2))+((pow(((0.5))*((h_inv)*((Ax(IDX_OF(N,a,b,N-2)))+((((-1))*(Ax(IDX_OF(N,a,b,0))))+((((-1))*(Az(IDX_OF(N,(a+N-1)%N,b,N-1))))+(Az(IDX_OF(N,(a+1)%N,b,N-1))))))),2))+((pow((((0.5))*((h_inv)*((At(IDX_OF(N,(a+N-1)%N,b,N-1)))+(((-1))*(At(IDX_OF(N,(a+1)%N,b,N-1)))))))+(dt_Ax(IDX_OF(N,a,b,N-1))),2))+((pow((((0.5))*((h_inv)*((At(IDX_OF(N,a,(b+N-1)%N,N-1)))+(((-1))*(At(IDX_OF(N,a,(b+1)%N,N-1)))))))+(dt_Ay(IDX_OF(N,a,b,N-1))),2))+(pow((((0.5))*((h_inv)*((At(IDX_OF(N,a,b,N-2)))+(((-1))*(At(IDX_OF(N,a,b,0)))))))+(dt_Az(IDX_OF(N,a,b,N-1))),2))))))));
	
      // rho(seqN(IDX_OF(N, a, b, 0), N)) = 0.5 *
      // 	( workspace.state(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).cwiseAbs2()
      // 	  + m * m * workspace.state(seqN(IDX_OF(N, a, b, 0), N)).cwiseAbs2()
      // 	  + 0.25 * inv_h_sqr *
      // 	  ( (workspace.state(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
      // 	     - workspace.state(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))).cwiseAbs2()
      // 	    + (workspace.state(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
      // 	       - workspace.state(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N))).cwiseAbs2() )
      // 	  );
      // rho(seqN(IDX_OF(N, a, b, 1), N-2)) += 0.5 * 0.25 * inv_h_sqr *
      // 	(workspace.state(seqN(IDX_OF(N, a, b, 2), N-2))
      // 	 - workspace.state(seqN(IDX_OF(N, a, b, 0), N-2))).cwiseAbs2();
      // rho(IDX_OF(N, a, b, 0)) += 0.5 * 0.25 * inv_h_sqr *
      // 	pow(workspace.state(IDX_OF(N, a, b, 1)) - workspace.state(IDX_OF(N, a, b, N-1)), 2);
      // rho(IDX_OF(N, a, b, N-1)) += 0.5 * 0.25 * inv_h_sqr *
      // 	pow(workspace.state(IDX_OF(N, a, b, 0)) - workspace.state(IDX_OF(N, a, b, N-2)), 2);
    }
  }
  return rho;
}


ProcaEquation::Vector ProcaEquation::compute_momentum_density(Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double h_inv = N / L;
  //  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  // const double inv_two_h = 1.0 / (2.0 * L / N);
  const long long int lattice_size = N * N * N;

  VectorXd At = compute_At(workspace, t);
  auto Ax = workspace.state.segment(0, lattice_size);
  auto Ay = workspace.state.segment(lattice_size, lattice_size);
  auto Az = workspace.state.segment(2 * lattice_size, lattice_size);
  auto dt_Ax = workspace.state.segment(3 * lattice_size, lattice_size);
  auto dt_Ay = workspace.state.segment(4 * lattice_size, lattice_size);
  auto dt_Az = workspace.state.segment(5 * lattice_size, lattice_size);
  
  VectorXd q(3 * lattice_size);
  auto q_x = q.segment(0, lattice_size);
  auto q_y = q.segment(lattice_size, lattice_size);
  auto q_z = q.segment(2 * lattice_size, lattice_size);
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
q_x.segment(IDX_OF(N, a, b, 1), N-2).array() = (((-1))*((pow(m,(2)))*((At.segment(IDX_OF(N,a,b,1),N-2).array())*(Ax.segment(IDX_OF(N,a,b,1),N-2).array()))))+((((-0.25))*((h_inv)*(((Ax.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2).array())+((((-1))*(Ax.segment(IDX_OF(N,a,(b+1)%N,1),N-2).array()))+((((-1))*(Ay.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2).array()))+(Ay.segment(IDX_OF(N,(a+1)%N,b,1),N-2).array()))))*(((h_inv)*(At.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2).array()))+((((-1))*((h_inv)*(At.segment(IDX_OF(N,a,(b+1)%N,1),N-2).array())))+(((2))*(dt_Ay.segment(IDX_OF(N,a,b,1),N-2).array())))))))+(((-0.25))*((h_inv)*(((Ax.segment(IDX_OF(N,a,b,0),N-2).array())+((((-1))*(Ax.segment(IDX_OF(N,a,b,2),N-2).array()))+((((-1))*(Az.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2).array()))+(Az.segment(IDX_OF(N,(a+1)%N,b,1),N-2).array()))))*(((h_inv)*(At.segment(IDX_OF(N,a,b,0),N-2).array()))+((((-1))*((h_inv)*(At.segment(IDX_OF(N,a,b,2),N-2).array())))+(((2))*(dt_Az.segment(IDX_OF(N,a,b,1),N-2).array()))))))));
q_x(IDX_OF(N, a, b, 0)) = (((-1))*((pow(m,(2)))*((At(IDX_OF(N,a,b,0)))*(Ax(IDX_OF(N,a,b,0))))))+((((-0.25))*((h_inv)*(((Ax(IDX_OF(N,a,(b+N-1)%N,0)))+((((-1))*(Ax(IDX_OF(N,a,(b+1)%N,0))))+((((-1))*(Ay(IDX_OF(N,(a+N-1)%N,b,0))))+(Ay(IDX_OF(N,(a+1)%N,b,0))))))*(((h_inv)*(At(IDX_OF(N,a,(b+N-1)%N,0))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,(b+1)%N,0)))))+(((2))*(dt_Ay(IDX_OF(N,a,b,0)))))))))+(((-0.25))*((h_inv)*(((Ax(IDX_OF(N,a,b,N-1)))+((((-1))*(Ax(IDX_OF(N,a,b,1))))+((((-1))*(Az(IDX_OF(N,(a+N-1)%N,b,0))))+(Az(IDX_OF(N,(a+1)%N,b,0))))))*(((h_inv)*(At(IDX_OF(N,a,b,N-1))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,b,1)))))+(((2))*(dt_Az(IDX_OF(N,a,b,0))))))))));
q_x(IDX_OF(N, a, b, N-1)) = (((-1))*((pow(m,(2)))*((At(IDX_OF(N,a,b,N-1)))*(Ax(IDX_OF(N,a,b,N-1))))))+((((-0.25))*((h_inv)*(((Ax(IDX_OF(N,a,(b+N-1)%N,N-1)))+((((-1))*(Ax(IDX_OF(N,a,(b+1)%N,N-1))))+((((-1))*(Ay(IDX_OF(N,(a+N-1)%N,b,N-1))))+(Ay(IDX_OF(N,(a+1)%N,b,N-1))))))*(((h_inv)*(At(IDX_OF(N,a,(b+N-1)%N,N-1))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,(b+1)%N,N-1)))))+(((2))*(dt_Ay(IDX_OF(N,a,b,N-1)))))))))+(((-0.25))*((h_inv)*(((Ax(IDX_OF(N,a,b,N-2)))+((((-1))*(Ax(IDX_OF(N,a,b,0))))+((((-1))*(Az(IDX_OF(N,(a+N-1)%N,b,N-1))))+(Az(IDX_OF(N,(a+1)%N,b,N-1))))))*(((h_inv)*(At(IDX_OF(N,a,b,N-2))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,b,0)))))+(((2))*(dt_Az(IDX_OF(N,a,b,N-1))))))))));
q_y.segment(IDX_OF(N, a, b, 1), N-2).array() = ((0.25))*((((-4))*((pow(m,(2)))*((At.segment(IDX_OF(N,a,b,1),N-2).array())*(Ay.segment(IDX_OF(N,a,b,1),N-2).array()))))+(((h_inv)*(((Ax.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2).array())+((((-1))*(Ax.segment(IDX_OF(N,a,(b+1)%N,1),N-2).array()))+((((-1))*(Ay.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2).array()))+(Ay.segment(IDX_OF(N,(a+1)%N,b,1),N-2).array()))))*(((h_inv)*(At.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2).array()))+((((-1))*((h_inv)*(At.segment(IDX_OF(N,(a+1)%N,b,1),N-2).array())))+(((2))*(dt_Ax.segment(IDX_OF(N,a,b,1),N-2).array()))))))+(((-1))*((h_inv)*(((Ay.segment(IDX_OF(N,a,b,0),N-2).array())+((((-1))*(Ay.segment(IDX_OF(N,a,b,2),N-2).array()))+((((-1))*(Az.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2).array()))+(Az.segment(IDX_OF(N,a,(b+1)%N,1),N-2).array()))))*(((h_inv)*(At.segment(IDX_OF(N,a,b,0),N-2).array()))+((((-1))*((h_inv)*(At.segment(IDX_OF(N,a,b,2),N-2).array())))+(((2))*(dt_Az.segment(IDX_OF(N,a,b,1),N-2).array())))))))));
q_y(IDX_OF(N, a, b, 0)) = ((0.25))*((((-4))*((pow(m,(2)))*((At(IDX_OF(N,a,b,0)))*(Ay(IDX_OF(N,a,b,0))))))+(((h_inv)*(((Ax(IDX_OF(N,a,(b+N-1)%N,0)))+((((-1))*(Ax(IDX_OF(N,a,(b+1)%N,0))))+((((-1))*(Ay(IDX_OF(N,(a+N-1)%N,b,0))))+(Ay(IDX_OF(N,(a+1)%N,b,0))))))*(((h_inv)*(At(IDX_OF(N,(a+N-1)%N,b,0))))+((((-1))*((h_inv)*(At(IDX_OF(N,(a+1)%N,b,0)))))+(((2))*(dt_Ax(IDX_OF(N,a,b,0))))))))+(((-1))*((h_inv)*(((Ay(IDX_OF(N,a,b,N-1)))+((((-1))*(Ay(IDX_OF(N,a,b,1))))+((((-1))*(Az(IDX_OF(N,a,(b+N-1)%N,0))))+(Az(IDX_OF(N,a,(b+1)%N,0))))))*(((h_inv)*(At(IDX_OF(N,a,b,N-1))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,b,1)))))+(((2))*(dt_Az(IDX_OF(N,a,b,0)))))))))));
q_y(IDX_OF(N, a, b, N-1)) = ((0.25))*((((-4))*((pow(m,(2)))*((At(IDX_OF(N,a,b,N-1)))*(Ay(IDX_OF(N,a,b,N-1))))))+(((h_inv)*(((Ax(IDX_OF(N,a,(b+N-1)%N,N-1)))+((((-1))*(Ax(IDX_OF(N,a,(b+1)%N,N-1))))+((((-1))*(Ay(IDX_OF(N,(a+N-1)%N,b,N-1))))+(Ay(IDX_OF(N,(a+1)%N,b,N-1))))))*(((h_inv)*(At(IDX_OF(N,(a+N-1)%N,b,N-1))))+((((-1))*((h_inv)*(At(IDX_OF(N,(a+1)%N,b,N-1)))))+(((2))*(dt_Ax(IDX_OF(N,a,b,N-1))))))))+(((-1))*((h_inv)*(((Ay(IDX_OF(N,a,b,N-2)))+((((-1))*(Ay(IDX_OF(N,a,b,0))))+((((-1))*(Az(IDX_OF(N,a,(b+N-1)%N,N-1))))+(Az(IDX_OF(N,a,(b+1)%N,N-1))))))*(((h_inv)*(At(IDX_OF(N,a,b,N-2))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,b,0)))))+(((2))*(dt_Az(IDX_OF(N,a,b,N-1)))))))))));
q_z.segment(IDX_OF(N, a, b, 1), N-2).array() = ((0.25))*((((-4))*((pow(m,(2)))*((At.segment(IDX_OF(N,a,b,1),N-2).array())*(Az.segment(IDX_OF(N,a,b,1),N-2).array()))))+(((h_inv)*(((Ax.segment(IDX_OF(N,a,b,0),N-2).array())+((((-1))*(Ax.segment(IDX_OF(N,a,b,2),N-2).array()))+((((-1))*(Az.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2).array()))+(Az.segment(IDX_OF(N,(a+1)%N,b,1),N-2).array()))))*(((h_inv)*(At.segment(IDX_OF(N,(a+N-1)%N,b,1),N-2).array()))+((((-1))*((h_inv)*(At.segment(IDX_OF(N,(a+1)%N,b,1),N-2).array())))+(((2))*(dt_Ax.segment(IDX_OF(N,a,b,1),N-2).array()))))))+((h_inv)*(((Ay.segment(IDX_OF(N,a,b,0),N-2).array())+((((-1))*(Ay.segment(IDX_OF(N,a,b,2),N-2).array()))+((((-1))*(Az.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2).array()))+(Az.segment(IDX_OF(N,a,(b+1)%N,1),N-2).array()))))*(((h_inv)*(At.segment(IDX_OF(N,a,(b+N-1)%N,1),N-2).array()))+((((-1))*((h_inv)*(At.segment(IDX_OF(N,a,(b+1)%N,1),N-2).array())))+(((2))*(dt_Ay.segment(IDX_OF(N,a,b,1),N-2).array()))))))));
q_z(IDX_OF(N, a, b, 0)) = ((0.25))*((((-4))*((pow(m,(2)))*((At(IDX_OF(N,a,b,0)))*(Az(IDX_OF(N,a,b,0))))))+(((h_inv)*(((Ax(IDX_OF(N,a,b,N-1)))+((((-1))*(Ax(IDX_OF(N,a,b,1))))+((((-1))*(Az(IDX_OF(N,(a+N-1)%N,b,0))))+(Az(IDX_OF(N,(a+1)%N,b,0))))))*(((h_inv)*(At(IDX_OF(N,(a+N-1)%N,b,0))))+((((-1))*((h_inv)*(At(IDX_OF(N,(a+1)%N,b,0)))))+(((2))*(dt_Ax(IDX_OF(N,a,b,0))))))))+((h_inv)*(((Ay(IDX_OF(N,a,b,N-1)))+((((-1))*(Ay(IDX_OF(N,a,b,1))))+((((-1))*(Az(IDX_OF(N,a,(b+N-1)%N,0))))+(Az(IDX_OF(N,a,(b+1)%N,0))))))*(((h_inv)*(At(IDX_OF(N,a,(b+N-1)%N,0))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,(b+1)%N,0)))))+(((2))*(dt_Ay(IDX_OF(N,a,b,0))))))))));
q_z(IDX_OF(N, a, b, N-1)) = ((0.25))*((((-4))*((pow(m,(2)))*((At(IDX_OF(N,a,b,N-1)))*(Az(IDX_OF(N,a,b,N-1))))))+(((h_inv)*(((Ax(IDX_OF(N,a,b,N-2)))+((((-1))*(Ax(IDX_OF(N,a,b,0))))+((((-1))*(Az(IDX_OF(N,(a+N-1)%N,b,N-1))))+(Az(IDX_OF(N,(a+1)%N,b,N-1))))))*(((h_inv)*(At(IDX_OF(N,(a+N-1)%N,b,N-1))))+((((-1))*((h_inv)*(At(IDX_OF(N,(a+1)%N,b,N-1)))))+(((2))*(dt_Ax(IDX_OF(N,a,b,N-1))))))))+((h_inv)*(((Ay(IDX_OF(N,a,b,N-2)))+((((-1))*(Ay(IDX_OF(N,a,b,0))))+((((-1))*(Az(IDX_OF(N,a,(b+N-1)%N,N-1))))+(Az(IDX_OF(N,a,(b+1)%N,N-1))))))*(((h_inv)*(At(IDX_OF(N,a,(b+N-1)%N,N-1))))+((((-1))*((h_inv)*(At(IDX_OF(N,a,(b+1)%N,N-1)))))+(((2))*(dt_Ay(IDX_OF(N,a,b,N-1))))))))));

      // q(seqN(IDX_OF(N, a, b, 0), N)).array() = - dt_varphi(seqN(IDX_OF(N, a, b, 0), N)).array() * inv_two_h
      // 	* ( varphi(seqN(IDX_OF(N, (a+1)%N, b, 0), N)) - varphi(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N)) ).array();

      // q(seqN(field_size + IDX_OF(N, a, b, 0), N)).array() = - dt_varphi(seqN(IDX_OF(N, a, b, 0), N)).array() * inv_two_h
      // 	* ( varphi(seqN(IDX_OF(N, a, (b+1)%N, 0), N)) - varphi(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N)) ).array();

      // q(seqN(2*field_size + IDX_OF(N, a, b, 1), N-2)).array() = - dt_varphi(seqN(IDX_OF(N, a, b, 1), N-2)).array() * inv_two_h
      // 	* ( varphi(seqN(IDX_OF(N, a, b, 2), N-2)) - varphi(seqN(IDX_OF(N, a, b, 0), N-2)) ).array();

      // q(2*field_size + IDX_OF(N, a, b, 0)) = - dt_varphi(IDX_OF(N, a, b, 0)) * inv_two_h
      // 	* ( varphi(IDX_OF(N, a, b, 1)) - varphi(IDX_OF(N, a, b, N-1)) );

      // q(2*field_size + IDX_OF(N, a, b, N-1)) = - dt_varphi(IDX_OF(N, a, b, N-1)) * inv_two_h
      // 	* ( varphi(IDX_OF(N, a, b, 0)) - varphi(IDX_OF(N, a, b, N-2)) );
      
    }
  }
  return q;
}


void check_proca_q(void)
{
  using namespace std::numbers;
  // Set the PRNG seed.
  RandomNormal::set_generator_seed(0);

  // Set the directory for output.
  const std::string dir = "/media/hypermania/Drive_001/FreeStreamingULDM/proca_IC/";
  prepare_directory_for_output(dir);
  
  // Set parameters for the simulation.
  MyParam param
    {
      .N = 384, // Lattice points per axis
      .L = 384 * 0.05, // Size of the box
      // ULDM params
      .m = 1.0, // Mass of scalar field
      .lambda = 0, // Lambda phi^4 coupling strength
      //.f_a = 30.0, // Not relevant for ComovingCurvatureEquationInFRW
      .k_ast = 5.0, // Characteristic momentum
      .k_Psi = 1.0, // Not relevant for ComovingCurvatureEquationInFRW
      .varphi_std_dev = 1.0, // Standard deviation of field
      .Psi_std_dev = 0.1, // Standard deviation of metric perturbation Psi
      // FRW metric params
      .a1 = 1.0,
      .H1 = 0.05,
      .t1 = 1.0 / (2 * param.H1),
      // Start and end time for numerical integration, and time interval between saves
      .t_start = param.t1,
      .t_end = param.t_start + (pow(3.5 / param.a1, 2) - 1.0) / (2 * param.H1),
      .t_interval = 49.99, // Save a snapshot every t_interval
      // Numerical method parameter
      .delta_t = 0.5, // Time step for numerical integration
      // Psi approximation parameter
      .M = 128 // Lattice points for storing / computing Psi
    };
  print_param(param);
  save_param_for_Mathematica(param, dir);

  
  typedef ProcaEquation Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;

  const long long int N = param.N;
  const long long int lattice_size = N*N*N;

  // Eigen::VectorXd rho_spectrum = load_VectorXd_from_file(dir + "rho_spectrum_old.dat");
  // const double rho_bar = sqrt(rho_spectrum[0] / (lattice_size * lattice_size));
  
  Eigen::VectorXd diff_q_0 = load_VectorXd_from_file(dir + "diff_q_0.dat");
  Eigen::VectorXd diff_q_1 = load_VectorXd_from_file(dir + "diff_q_1.dat");
  Eigen::VectorXd diff_q_2 = load_VectorXd_from_file(dir + "diff_q_2.dat");
  
  std::cout << "point 0 \n";


  auto fft_wrapper = fftwWrapper(N);
  Eigen::VectorXd diff_q_spectrum(3*(N/2)*(N/2)+1);
  diff_q_spectrum.array() = 0;
  
  diff_q_spectrum += compute_power_spectrum(N, diff_q_0, fft_wrapper);
  diff_q_spectrum += compute_power_spectrum(N, diff_q_1, fft_wrapper);
  diff_q_spectrum += compute_power_spectrum(N, diff_q_2, fft_wrapper);
  write_to_file(diff_q_spectrum, dir + "diff_q_spectrum.dat");
  
}
