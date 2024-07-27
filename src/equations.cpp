#include "equations.hpp"

#include "fdm3d.hpp"
#include "physics.hpp"



void KleinGordonEquation::operator()(const State &x, State &dxdt, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      dxdt(seqN(IDX_OF(N, a, b, 0), N)) = x(seqN(N*N*N+IDX_OF(N, a, b, 0), N));
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 0), N)) =
	(-1.0) * m * m * x(seqN(IDX_OF(N, a, b, 0), N))
	+ inv_h_sqr * (-6.0 * x(seqN(IDX_OF(N, a, b, 0), N))
		       + x(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
		       + x(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))
		       + x(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
		       + x(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N)));
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 1), N-2)) +=
	inv_h_sqr * ( x(seqN(IDX_OF(N, a, b, 2), N-2))
		      + x(seqN(IDX_OF(N, a, b, 0), N-2)) );
      dxdt(N*N*N+IDX_OF(N, a, b, 0)) +=
	inv_h_sqr * ( x(IDX_OF(N, a, b, N-1)) + x(IDX_OF(N, a, b, 1)) );
      dxdt(N*N*N+IDX_OF(N, a, b, N-1)) +=
	inv_h_sqr * ( x(IDX_OF(N, a, b, N-2)) + x(IDX_OF(N, a, b, 0)) );
    }
  }

}

KleinGordonEquation::Vector KleinGordonEquation::compute_energy_density(const Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  
  VectorXd rho(workspace.state.size() / 2);
    
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

void KleinGordonEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  //const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      dxdt(seqN(IDX_OF(N, a, b, 0), N)) = x(seqN(N*N*N+IDX_OF(N, a, b, 0), N));
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 0), N)) =
  	(-3.0) * H_t * x(seqN(N*N*N+IDX_OF(N, a, b, 0), N))
  	- m * m * x(seqN(IDX_OF(N, a, b, 0), N))
  	+ inv_ah_sqr * (-6.0 * x(seqN(IDX_OF(N, a, b, 0), N))
  			+ x(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
  			+ x(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))
  			+ x(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
  			+ x(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N)));
	  
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 1), N-2)) +=
  	inv_ah_sqr * ( x(seqN(IDX_OF(N, a, b, 2), N-2))
  		       + x(seqN(IDX_OF(N, a, b, 0), N-2)) );
      dxdt(N*N*N+IDX_OF(N, a, b, 0)) +=
  	inv_ah_sqr * ( x(IDX_OF(N, a, b, N-1)) + x(IDX_OF(N, a, b, 1)) );
      dxdt(N*N*N+IDX_OF(N, a, b, N-1)) +=
  	inv_ah_sqr * ( x(IDX_OF(N, a, b, N-2)) + x(IDX_OF(N, a, b, 0)) );
    }
  }

}

KleinGordonEquationInFRW::Vector KleinGordonEquationInFRW::compute_energy_density(const Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  
  VectorXd rho(workspace.state.size() / 2);
    
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      rho(seqN(IDX_OF(N, a, b, 0), N)) = 0.5 *
	( workspace.state(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).cwiseAbs2()
	  + m * m * workspace.state(seqN(IDX_OF(N, a, b, 0), N)).cwiseAbs2()
	  + 0.25 * inv_ah_sqr *
	  ( (workspace.state(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
	     - workspace.state(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))).cwiseAbs2()
	    + (workspace.state(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
	       - workspace.state(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N))).cwiseAbs2() )
	  );
      rho(seqN(IDX_OF(N, a, b, 1), N-2)) += 0.5 * 0.25 * inv_ah_sqr *
	(workspace.state(seqN(IDX_OF(N, a, b, 2), N-2))
	 - workspace.state(seqN(IDX_OF(N, a, b, 0), N-2))).cwiseAbs2();
      rho(IDX_OF(N, a, b, 0)) += 0.5 * 0.25 * inv_ah_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 1)) - workspace.state(IDX_OF(N, a, b, N-1)), 2);
      rho(IDX_OF(N, a, b, N-1)) += 0.5 * 0.25 * inv_ah_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 0)) - workspace.state(IDX_OF(N, a, b, N-2)), 2);
    }
  }
  return rho;
}


void compute_Comoving_Curvature_Psi_dPsidt_fft(const Eigen::VectorXd &R_fft, Eigen::VectorXd &Psi_fft, Eigen::VectorXd &dPsidt_fft, const long long int N, const double factor, const double H_t)
{
  using namespace std::numbers;
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      for(long long int c = 0; c <= N/2; ++c){
	long long int a_shifted = (a<=N/2) ? a : (N-a);
	long long int b_shifted = (b<=N/2) ? b : (N-b);
	long long int c_shifted = (c<=N/2) ? c : (N-c);
	long long int s_sqr = a_shifted*a_shifted + b_shifted*b_shifted + c_shifted*c_shifted;
	long long int idx = N*(N/2+1)*a + (N/2+1)*b + c_shifted;
	long long int offset_k = 2 * idx;

	if(s_sqr == 0) {
	  Psi_fft(offset_k) = 0;
	  Psi_fft(offset_k + 1) = 0;
	  dPsidt_fft(offset_k) = 0;
	  dPsidt_fft(offset_k + 1) = 0;
	} else {
	  double omega_eta = sqrt(static_cast<double>(s_sqr)) * factor;
	  //double sin_val = sin(omega_eta);
	  //double cos_val = cos(omega_eta);
	  double sin_val;
	  double cos_val;
	  sincos(omega_eta, &sin_val, &cos_val);
	  double common_factor = 2.0 / (omega_eta * omega_eta * omega_eta) / (N * N * N);

	  double R_val_r = R_fft(offset_k);
	  double R_val_i = R_fft(offset_k + 1);

	  double transfer_function_Psi = (sin_val - omega_eta * cos_val) * common_factor;
	  Psi_fft(offset_k) = R_val_r * transfer_function_Psi;
	  Psi_fft(offset_k + 1) = R_val_i * transfer_function_Psi;

	  double transfer_function_dPsidt = H_t * (3 * omega_eta * cos_val + (omega_eta * omega_eta - 3.0) * sin_val) * common_factor;
	  dPsidt_fft(offset_k) = R_val_r * transfer_function_dPsidt;
	  dPsidt_fft(offset_k + 1) = R_val_i * transfer_function_dPsidt;
	}
      }
    }
  }
}


void ComovingCurvatureEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  using namespace Eigen;
  using namespace std::numbers;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  const double factor = (2 * pi / L) * eta_t / sqrt(3);
  
  // Eigen::VectorXd Psi_fft(workspace.R_fft.size());
  // Eigen::VectorXd dPsidt_fft(workspace.R_fft.size());
  // compute_Comoving_Curvature_Psi_dPsidt_fft(workspace.R_fft, Psi_fft, dPsidt_fft, N, factor, H_t);
  // Eigen::VectorXd Psi = workspace.fft_wrapper.execute_z2d(Psi_fft);
  // Eigen::VectorXd dPsidt = workspace.fft_wrapper.execute_z2d(dPsidt_fft);

  // static Eigen::VectorXd Psi_fft(workspace.R_fft.size());
  // static Eigen::VectorXd dPsidt_fft(workspace.R_fft.size());
  // static Eigen::VectorXd Psi(N * N * N);
  // static Eigen::VectorXd dPsidt(N * N * N);

  // compute_Comoving_Curvature_Psi_dPsidt_fft(workspace.R_fft, Psi_fft, dPsidt_fft, N, factor, H_t);

  // workspace.fft_wrapper.execute_z2d(Psi_fft, Psi);
  // workspace.fft_wrapper.execute_z2d(dPsidt_fft, dPsidt);

  auto &Psi = workspace.Psi;
  auto &dPsidt = workspace.dPsidt;
  auto &Psi_fft = workspace.Psi_fft;
  auto &dPsidt_fft = workspace.dPsidt_fft;
  if(Psi_fft.size() != workspace.R_fft.size()
     || dPsidt_fft.size() != workspace.R_fft.size()
     || Psi.size() != N * N * N
     || dPsidt.size() != N * N * N)
    {
      Psi_fft.resize(workspace.R_fft.size());
      dPsidt_fft.resize(workspace.R_fft.size());
      Psi.resize(N * N * N);
      dPsidt.resize(N * N * N);
    }
  
  compute_Comoving_Curvature_Psi_dPsidt_fft(workspace.R_fft, Psi_fft, dPsidt_fft, N, factor, H_t);

  workspace.fft_wrapper.execute_z2d(Psi_fft, Psi);
  workspace.fft_wrapper.execute_z2d(dPsidt_fft, dPsidt);
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      dxdt(seqN(IDX_OF(N, a, b, 0), N)) = x(seqN(N*N*N+IDX_OF(N, a, b, 0), N));
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).array() =
  	(-3.0 * H_t + 4.0 * dPsidt(seqN(IDX_OF(N, a, b, 0), N)).array()) * x(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).array()
  	- exp(2 * Psi(seqN(IDX_OF(N, a, b, 0), N)).array()) * m * m * x(seqN(IDX_OF(N, a, b, 0), N)).array()
  	+ exp(4 * Psi(seqN(IDX_OF(N, a, b, 0), N)).array()) * inv_ah_sqr
	* (-6.0 * x(seqN(IDX_OF(N, a, b, 0), N)).array()
	   + x(seqN(IDX_OF(N, (a+1)%N, b, 0), N)).array()
	   + x(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N)).array()
	   + x(seqN(IDX_OF(N, a, (b+1)%N, 0), N)).array()
	   + x(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N)).array() );
	  
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 1), N-2)).array() +=
  	exp(4 * Psi(seqN(IDX_OF(N, a, b, 1), N-2)).array()) * inv_ah_sqr
	* ( x(seqN(IDX_OF(N, a, b, 2), N-2)).array()
	    + x(seqN(IDX_OF(N, a, b, 0), N-2)).array() );
      dxdt(N*N*N+IDX_OF(N, a, b, 0)) +=
  	exp(4 * Psi(IDX_OF(N, a, b, 0))) * inv_ah_sqr
	* ( x(IDX_OF(N, a, b, N-1)) + x(IDX_OF(N, a, b, 1)) );
      dxdt(N*N*N+IDX_OF(N, a, b, N-1)) +=
  	exp(4 * Psi(IDX_OF(N, a, b, N-1))) * inv_ah_sqr
	* ( x(IDX_OF(N, a, b, N-2)) + x(IDX_OF(N, a, b, 0)) );
    }
  }

}



ComovingCurvatureEquationInFRW::Vector ComovingCurvatureEquationInFRW::compute_energy_density(Workspace &workspace, const double t)
{
  using namespace Eigen;
  using namespace std::numbers;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  const double factor = (2 * pi / L) * eta_t / sqrt(3);
  
  // Eigen::VectorXd Psi(workspace.R_fft.size());
  // {
  //   Eigen::VectorXd dPsidt(workspace.R_fft.size());
  //   compute_Comoving_Curvature_Psi_dPsidt_fft(workspace.R_fft, Psi, dPsidt, N, factor, H_t);
  //   workspace.fft_wrapper.execute_inplace_z2d(Psi);

  //   IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  //   std::cout << "Psi = " << Psi.head(16).transpose().format(HeavyFmt) << '\n';
  // }

  Eigen::VectorXd Psi_fft(workspace.R_fft.size());
  {
    Eigen::VectorXd dPsidt_fft(workspace.R_fft.size());
    compute_Comoving_Curvature_Psi_dPsidt_fft(workspace.R_fft, Psi_fft, dPsidt_fft, N, factor, H_t);
  }
  auto Psi = workspace.fft_wrapper.execute_z2d(Psi_fft);
  // IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  // std::cout << "Psi = " << Psi.head(16).transpose().format(HeavyFmt) << '\n';

  
  VectorXd rho(workspace.state.size() / 2);
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      rho(seqN(IDX_OF(N, a, b, 0), N)).array() = 0.5 *
	( exp(-2 * Psi(seqN(IDX_OF(N, a, b, 0), N)).array()) * workspace.state(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).array().abs2()
	  + m * m * workspace.state(seqN(IDX_OF(N, a, b, 0), N)).array().abs2()
	  + exp(2 * Psi(seqN(IDX_OF(N, a, b, 0), N)).array()) * 0.25 * inv_ah_sqr *
	  ( (workspace.state(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
	     - workspace.state(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))).array().abs2()
	    + (workspace.state(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
	       - workspace.state(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N))).array().abs2() )
	  );
      rho(seqN(IDX_OF(N, a, b, 1), N-2)).array() +=
	exp(2 * Psi(seqN(IDX_OF(N, a, b, 1), N-2)).array()) * 0.5 * 0.25 * inv_ah_sqr *
	(workspace.state(seqN(IDX_OF(N, a, b, 2), N-2))
	 - workspace.state(seqN(IDX_OF(N, a, b, 0), N-2))).array().abs2();
      rho(IDX_OF(N, a, b, 0)) += exp(2 * Psi(IDX_OF(N, a, b, 0))) * 0.5 * 0.25 * inv_ah_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 1)) - workspace.state(IDX_OF(N, a, b, N-1)), 2);
      rho(IDX_OF(N, a, b, N-1)) += exp(2 * Psi(IDX_OF(N, a, b, N-1))) * 0.5 * 0.25 * inv_ah_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 0)) - workspace.state(IDX_OF(N, a, b, N-2)), 2);
    }
  }
  return rho;
}
