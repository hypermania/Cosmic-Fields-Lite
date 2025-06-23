#include "sp.hpp"

Eigen::ArrayXcd generate_gaussian_random_sp_field(const long long int N, const double L, const Spectrum &P)
{
  const long long int lattice_size = N*N*N;
  Eigen::VectorXd field_re(3 * lattice_size);
  Eigen::VectorXd field_im(3 * lattice_size);
  field_re.segment(0, lattice_size) = generate_gaussian_random_field(N, L, P);
  field_re.segment(lattice_size, lattice_size) = generate_gaussian_random_field(N, L, P);
  field_re.segment(2 * lattice_size, lattice_size) = generate_gaussian_random_field(N, L, P);
  field_im.segment(0, lattice_size) = generate_gaussian_random_field(N, L, P);
  field_im.segment(lattice_size, lattice_size) = generate_gaussian_random_field(N, L, P);
  field_im.segment(2 * lattice_size, lattice_size) = generate_gaussian_random_field(N, L, P);
  Eigen::VectorXcd field = field_re + std::complex<double>(0, 1) * field_im;
  return field;
}

Eigen::ArrayXcd generate_inhomogeneous_gaussian_random_sp_field(const long long int N, const double L, const Eigen::VectorXd &f, const Spectrum &P)
{
  const long long int lattice_size = N*N*N;
  Eigen::VectorXd field_re(3 * lattice_size);
  Eigen::VectorXd field_im(3 * lattice_size);
  field_re.segment(0, lattice_size) = generate_inhomogeneous_gaussian_random_field(N, L, f, P);
  field_re.segment(lattice_size, lattice_size) = generate_inhomogeneous_gaussian_random_field(N, L, f, P);
  field_re.segment(2 * lattice_size, lattice_size) = generate_inhomogeneous_gaussian_random_field(N, L, f, P);
  field_im.segment(0, lattice_size) = generate_inhomogeneous_gaussian_random_field(N, L, f, P);
  field_im.segment(lattice_size, lattice_size) = generate_inhomogeneous_gaussian_random_field(N, L, f, P);
  field_im.segment(2 * lattice_size, lattice_size) = generate_inhomogeneous_gaussian_random_field(N, L, f, P);
  Eigen::VectorXcd field = field_re + std::complex<double>(0, 1) * field_im;
  return field;
}


SchrodingerPoissonEquation::Vector SchrodingerPoissonEquation::compute_energy_density(Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  // const double L = workspace.L;
  const double m = workspace.m;
  // const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  // const double h_inv = N / L;
  const double a_t = workspace.cosmology.a(t);
  const long long int lattice_size = N*N*N;
  VectorXd rho(lattice_size);

  auto &state = workspace.state;
  auto &psi = workspace.Psi;
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      rho.segment(IDX_OF(N, a, b, 0), N).array() = (0.50000000000000000000)*(m)*(((state.segment(0*lattice_size+IDX_OF(N,a,b,0),N)).abs2())+((state.segment(1*lattice_size+IDX_OF(N,a,b,0),N)).abs2())+((state.segment(2*lattice_size+IDX_OF(N,a,b,0),N)).abs2()))*((1)+(exp((2)*(psi.segment(IDX_OF(N,a,b,0),N).array()))))*(pow(a_t,-3));
    }
  }
  return rho;
}


// TODO
SchrodingerPoissonEquation::Vector SchrodingerPoissonEquation::compute_momentum_density(Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  // const double m = workspace.m;
  const double h_inv = N / L;
  const double a_t = workspace.cosmology.a(t);
  //  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  const long long int lattice_size = N * N * N;

  VectorXd q(3 * lattice_size);
  auto q_x = q.segment(0, lattice_size);
  auto q_y = q.segment(lattice_size, lattice_size);
  auto q_z = q.segment(2 * lattice_size, lattice_size);

  auto &state = workspace.state;
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
q_x.segment(IDX_OF(N, a, b, 1), N-2).array() = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state.segment(0*lattice_size+IDX_OF(N,(a+N-1)%N,b,1),N-2)))+((-1.0000000000000000000)*(conj(state.segment(0*lattice_size+IDX_OF(N,(a+1)%N,b,1),N-2)))))*(state.segment(0*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(0*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(0*lattice_size+IDX_OF(N,(a+N-1)%N,b,1),N-2)))+(state.segment(0*lattice_size+IDX_OF(N,(a+1)%N,b,1),N-2))))+(((conj(state.segment(1*lattice_size+IDX_OF(N,(a+N-1)%N,b,1),N-2)))+((-1.0000000000000000000)*(conj(state.segment(1*lattice_size+IDX_OF(N,(a+1)%N,b,1),N-2)))))*(state.segment(1*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(1*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(1*lattice_size+IDX_OF(N,(a+N-1)%N,b,1),N-2)))+(state.segment(1*lattice_size+IDX_OF(N,(a+1)%N,b,1),N-2))))+(((conj(state.segment(2*lattice_size+IDX_OF(N,(a+N-1)%N,b,1),N-2)))+((-1.0000000000000000000)*(conj(state.segment(2*lattice_size+IDX_OF(N,(a+1)%N,b,1),N-2)))))*(state.segment(2*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(2*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(2*lattice_size+IDX_OF(N,(a+N-1)%N,b,1),N-2)))+(state.segment(2*lattice_size+IDX_OF(N,(a+1)%N,b,1),N-2))))));
q_x(IDX_OF(N, a, b, 0)) = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state(0*lattice_size+IDX_OF(N,(a+N-1)%N,b,0))))+((-1.0000000000000000000)*(conj(state(0*lattice_size+IDX_OF(N,(a+1)%N,b,0))))))*(state(0*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(0*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(0*lattice_size+IDX_OF(N,(a+N-1)%N,b,0))))+(state(0*lattice_size+IDX_OF(N,(a+1)%N,b,0)))))+(((conj(state(1*lattice_size+IDX_OF(N,(a+N-1)%N,b,0))))+((-1.0000000000000000000)*(conj(state(1*lattice_size+IDX_OF(N,(a+1)%N,b,0))))))*(state(1*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(1*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(1*lattice_size+IDX_OF(N,(a+N-1)%N,b,0))))+(state(1*lattice_size+IDX_OF(N,(a+1)%N,b,0)))))+(((conj(state(2*lattice_size+IDX_OF(N,(a+N-1)%N,b,0))))+((-1.0000000000000000000)*(conj(state(2*lattice_size+IDX_OF(N,(a+1)%N,b,0))))))*(state(2*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(2*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(2*lattice_size+IDX_OF(N,(a+N-1)%N,b,0))))+(state(2*lattice_size+IDX_OF(N,(a+1)%N,b,0)))))));
q_x(IDX_OF(N, a, b, N-1)) = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state(0*lattice_size+IDX_OF(N,(a+N-1)%N,b,N-1))))+((-1.0000000000000000000)*(conj(state(0*lattice_size+IDX_OF(N,(a+1)%N,b,N-1))))))*(state(0*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(0*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(0*lattice_size+IDX_OF(N,(a+N-1)%N,b,N-1))))+(state(0*lattice_size+IDX_OF(N,(a+1)%N,b,N-1)))))+(((conj(state(1*lattice_size+IDX_OF(N,(a+N-1)%N,b,N-1))))+((-1.0000000000000000000)*(conj(state(1*lattice_size+IDX_OF(N,(a+1)%N,b,N-1))))))*(state(1*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(1*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(1*lattice_size+IDX_OF(N,(a+N-1)%N,b,N-1))))+(state(1*lattice_size+IDX_OF(N,(a+1)%N,b,N-1)))))+(((conj(state(2*lattice_size+IDX_OF(N,(a+N-1)%N,b,N-1))))+((-1.0000000000000000000)*(conj(state(2*lattice_size+IDX_OF(N,(a+1)%N,b,N-1))))))*(state(2*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(2*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(2*lattice_size+IDX_OF(N,(a+N-1)%N,b,N-1))))+(state(2*lattice_size+IDX_OF(N,(a+1)%N,b,N-1)))))));
q_y.segment(IDX_OF(N, a, b, 1), N-2).array() = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state.segment(0*lattice_size+IDX_OF(N,a,(b+N-1)%N,1),N-2)))+((-1.0000000000000000000)*(conj(state.segment(0*lattice_size+IDX_OF(N,a,(b+1)%N,1),N-2)))))*(state.segment(0*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(0*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(0*lattice_size+IDX_OF(N,a,(b+N-1)%N,1),N-2)))+(state.segment(0*lattice_size+IDX_OF(N,a,(b+1)%N,1),N-2))))+(((conj(state.segment(1*lattice_size+IDX_OF(N,a,(b+N-1)%N,1),N-2)))+((-1.0000000000000000000)*(conj(state.segment(1*lattice_size+IDX_OF(N,a,(b+1)%N,1),N-2)))))*(state.segment(1*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(1*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(1*lattice_size+IDX_OF(N,a,(b+N-1)%N,1),N-2)))+(state.segment(1*lattice_size+IDX_OF(N,a,(b+1)%N,1),N-2))))+(((conj(state.segment(2*lattice_size+IDX_OF(N,a,(b+N-1)%N,1),N-2)))+((-1.0000000000000000000)*(conj(state.segment(2*lattice_size+IDX_OF(N,a,(b+1)%N,1),N-2)))))*(state.segment(2*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(2*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(2*lattice_size+IDX_OF(N,a,(b+N-1)%N,1),N-2)))+(state.segment(2*lattice_size+IDX_OF(N,a,(b+1)%N,1),N-2))))));
q_y(IDX_OF(N, a, b, 0)) = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state(0*lattice_size+IDX_OF(N,a,(b+N-1)%N,0))))+((-1.0000000000000000000)*(conj(state(0*lattice_size+IDX_OF(N,a,(b+1)%N,0))))))*(state(0*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(0*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(0*lattice_size+IDX_OF(N,a,(b+N-1)%N,0))))+(state(0*lattice_size+IDX_OF(N,a,(b+1)%N,0)))))+(((conj(state(1*lattice_size+IDX_OF(N,a,(b+N-1)%N,0))))+((-1.0000000000000000000)*(conj(state(1*lattice_size+IDX_OF(N,a,(b+1)%N,0))))))*(state(1*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(1*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(1*lattice_size+IDX_OF(N,a,(b+N-1)%N,0))))+(state(1*lattice_size+IDX_OF(N,a,(b+1)%N,0)))))+(((conj(state(2*lattice_size+IDX_OF(N,a,(b+N-1)%N,0))))+((-1.0000000000000000000)*(conj(state(2*lattice_size+IDX_OF(N,a,(b+1)%N,0))))))*(state(2*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(2*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(2*lattice_size+IDX_OF(N,a,(b+N-1)%N,0))))+(state(2*lattice_size+IDX_OF(N,a,(b+1)%N,0)))))));
q_y(IDX_OF(N, a, b, N-1)) = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state(0*lattice_size+IDX_OF(N,a,(b+N-1)%N,N-1))))+((-1.0000000000000000000)*(conj(state(0*lattice_size+IDX_OF(N,a,(b+1)%N,N-1))))))*(state(0*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(0*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(0*lattice_size+IDX_OF(N,a,(b+N-1)%N,N-1))))+(state(0*lattice_size+IDX_OF(N,a,(b+1)%N,N-1)))))+(((conj(state(1*lattice_size+IDX_OF(N,a,(b+N-1)%N,N-1))))+((-1.0000000000000000000)*(conj(state(1*lattice_size+IDX_OF(N,a,(b+1)%N,N-1))))))*(state(1*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(1*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(1*lattice_size+IDX_OF(N,a,(b+N-1)%N,N-1))))+(state(1*lattice_size+IDX_OF(N,a,(b+1)%N,N-1)))))+(((conj(state(2*lattice_size+IDX_OF(N,a,(b+N-1)%N,N-1))))+((-1.0000000000000000000)*(conj(state(2*lattice_size+IDX_OF(N,a,(b+1)%N,N-1))))))*(state(2*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(2*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(2*lattice_size+IDX_OF(N,a,(b+N-1)%N,N-1))))+(state(2*lattice_size+IDX_OF(N,a,(b+1)%N,N-1)))))));
q_z.segment(IDX_OF(N, a, b, 1), N-2).array() = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state.segment(0*lattice_size+IDX_OF(N,a,b,0),N-2)))+((-1.0000000000000000000)*(conj(state.segment(0*lattice_size+IDX_OF(N,a,b,2),N-2)))))*(state.segment(0*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(0*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(0*lattice_size+IDX_OF(N,a,b,0),N-2)))+(state.segment(0*lattice_size+IDX_OF(N,a,b,2),N-2))))+(((conj(state.segment(1*lattice_size+IDX_OF(N,a,b,0),N-2)))+((-1.0000000000000000000)*(conj(state.segment(1*lattice_size+IDX_OF(N,a,b,2),N-2)))))*(state.segment(1*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(1*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(1*lattice_size+IDX_OF(N,a,b,0),N-2)))+(state.segment(1*lattice_size+IDX_OF(N,a,b,2),N-2))))+(((conj(state.segment(2*lattice_size+IDX_OF(N,a,b,0),N-2)))+((-1.0000000000000000000)*(conj(state.segment(2*lattice_size+IDX_OF(N,a,b,2),N-2)))))*(state.segment(2*lattice_size+IDX_OF(N,a,b,1),N-2)))+((conj(state.segment(2*lattice_size+IDX_OF(N,a,b,1),N-2)))*(((-1.0000000000000000000)*(state.segment(2*lattice_size+IDX_OF(N,a,b,0),N-2)))+(state.segment(2*lattice_size+IDX_OF(N,a,b,2),N-2))))));
q_z(IDX_OF(N, a, b, 0)) = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state(0*lattice_size+IDX_OF(N,a,b,N-1))))+((-1.0000000000000000000)*(conj(state(0*lattice_size+IDX_OF(N,a,b,1))))))*(state(0*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(0*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(0*lattice_size+IDX_OF(N,a,b,N-1))))+(state(0*lattice_size+IDX_OF(N,a,b,1)))))+(((conj(state(1*lattice_size+IDX_OF(N,a,b,N-1))))+((-1.0000000000000000000)*(conj(state(1*lattice_size+IDX_OF(N,a,b,1))))))*(state(1*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(1*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(1*lattice_size+IDX_OF(N,a,b,N-1))))+(state(1*lattice_size+IDX_OF(N,a,b,1)))))+(((conj(state(2*lattice_size+IDX_OF(N,a,b,N-1))))+((-1.0000000000000000000)*(conj(state(2*lattice_size+IDX_OF(N,a,b,1))))))*(state(2*lattice_size+IDX_OF(N,a,b,0))))+((conj(state(2*lattice_size+IDX_OF(N,a,b,0))))*(((-1.0000000000000000000)*(state(2*lattice_size+IDX_OF(N,a,b,N-1))))+(state(2*lattice_size+IDX_OF(N,a,b,1)))))));
q_z(IDX_OF(N, a, b, N-1)) = real((std::complex<double>(0,-0.25000000000000000000))*(h_inv)*(pow(a_t,-4))*((((conj(state(0*lattice_size+IDX_OF(N,a,b,N-2))))+((-1.0000000000000000000)*(conj(state(0*lattice_size+IDX_OF(N,a,b,0))))))*(state(0*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(0*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(0*lattice_size+IDX_OF(N,a,b,N-2))))+(state(0*lattice_size+IDX_OF(N,a,b,0)))))+(((conj(state(1*lattice_size+IDX_OF(N,a,b,N-2))))+((-1.0000000000000000000)*(conj(state(1*lattice_size+IDX_OF(N,a,b,0))))))*(state(1*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(1*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(1*lattice_size+IDX_OF(N,a,b,N-2))))+(state(1*lattice_size+IDX_OF(N,a,b,0)))))+(((conj(state(2*lattice_size+IDX_OF(N,a,b,N-2))))+((-1.0000000000000000000)*(conj(state(2*lattice_size+IDX_OF(N,a,b,0))))))*(state(2*lattice_size+IDX_OF(N,a,b,N-1))))+((conj(state(2*lattice_size+IDX_OF(N,a,b,N-1))))*(((-1.0000000000000000000)*(state(2*lattice_size+IDX_OF(N,a,b,N-2))))+(state(2*lattice_size+IDX_OF(N,a,b,0)))))));
          }
  }
  return q;
}


