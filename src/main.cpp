#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <filesystem>

#include "Eigen/Dense"

#include "param.hpp"
#include "initializer.hpp"
#include "random_field.hpp"
#include "io.hpp"
#include "fdm3d.hpp"
#include "utility.hpp"
#include "physics.hpp"
#include "equations.hpp"
#include "workspace.hpp"
#include "observer.hpp"
#include "midpoint.hpp"
#include "wkb.hpp"
#include "field_booster.hpp"
#include "proca.hpp"
#include "sp.hpp"

#ifndef DISABLE_CUDA
#include <thrust/device_vector.h>
#include "equations_cuda.cuh"
#include "cufft.h"
#include "fdm3d_cuda.cuh"
#endif

// The struct containing parameters for the simulation.
// You can add new params for other simulations.
struct MyParam {
  // lattice params
  long long int N;
  double L;
  // ULDM params
  double m;
  double lambda;
  double f_a;
  double k_ast;
  double k_Psi;
  double varphi_std_dev;
  double Psi_std_dev;
  // FRW metric params
  double a1;
  double H1;
  double t1;
  // Solution record params
  double t_start;
  double t_end;
  double t_interval;
  // Numerical method parameter
  double delta_t;
  // Psi approximation parameter (the size of the grid storing Psi)
  long long int M;
  // Params for adding fluctuations on a homogeneous background
  double f;
  double delta_varphi_std_dev;
  double k_delta_varphi;
};

void solve_field_equation(void);
void generate_wkb_solutions(void);
void generate_ic_kg(void);
void generate_ic_proca(void);
void generate_ic_sp(void);
void check_proca_q(void);

int main(int argc, char **argv){
  // Runs the simulation described in Section 4.2.2 of paper.
  
  // Solve scalar field equation in a background of comoving curvature perturbation.
  // Save output to output/Growth_and_FS/
  // solve_field_equation();

  // Optional: Use WKB solution to extend the simulation.
  //generate_wkb_solutions();

  // generate_ic_kg();
  generate_ic_proca();
  // check_proca_q();
  // generate_ic_sp();

  // Eigen::VectorXd v(10);
  // Eigen::VectorXd w(10);
  // v.array() = 100;
  // w.array() = 200;
  // std::cout << "v.size() = " << v.size() << std::endl;
  // std::cout << "w.size() = " << w.size() << std::endl;
  // Eigen::VectorXd().swap(v);
  // std::cout << "v.size() = " << v.size() << std::endl;
  // std::cout << "w.size() = " << w.size() << std::endl;
  // v = w;
  // std::cout << "v.size() = " << v.size() << std::endl;
  // std::cout << "w.size() = " << w.size() << std::endl;
}  

void generate_ic_kg(void)
{
  // Set the PRNG seed.
  RandomNormal::set_generator_seed(0);

  
  // Set the directory for output.
  // const std::string dir = "output/scalar_IC/";
  const std::string dir = "/media/hypermania/Drive_001/FreeStreamingULDM/scalar_IC/";
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
      .Psi_std_dev = 0.2, // Standard deviation of metric perturbation Psi
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

  
  typedef KleinGordonEquation Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;


  
  const long long int N = param.N;
  Eigen::VectorXd tau(N*N*N);
  // Spectrum P_tau = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
  // Eigen::VectorXd tau = generate_gaussian_random_field(param.N, param.L, P_tau);

  for(int a = 0; a < N; ++a){
    for(int b = 0; b < N; ++b){
      for(int c = 0; c < N; ++c){
	tau(IDX_OF(N, a, b, c)) = -0.5 * cos(2 * std::numbers::pi * c / N);
      }
    }
  }
  write_to_file(tau, dir + "tau.dat");

  Workspace workspace(param, unperturbed_grf);
  
  {
    Eigen::VectorXd varphi_old = workspace.state.head(N*N*N);
    Eigen::VectorXd dt_varphi_old = workspace.state.tail(N*N*N);
    write_to_file(varphi_old, dir + "varphi_old.dat");
    write_to_file(dt_varphi_old, dir + "dt_varphi_old.dat");
  }

  {
    Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, 0);
    write_to_file(compute_mode_power_spectrum(N, param.L, param.m, 1.0, workspace.state, workspace.fft_wrapper), dir + "varphi_spectrum_old.dat");
    write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum_old.dat");
    write_to_file(rho_old, dir + "rho_old.dat");
  }
  
  {
    Eigen::VectorXd q_old = Equation::compute_momentum_density(workspace, 0);
    const long long int field_size = N*N*N;
    Eigen::VectorXd q_spectrum(3*(N/2)*(N/2)+1);
    q_spectrum.array() = 0;
    for(size_t idx = 0; idx < 3; ++idx){
      Eigen::VectorXd q_idx = q_old.segment(idx * field_size, field_size);
      q_spectrum += compute_power_spectrum(N, q_idx, workspace.fft_wrapper);
    }
    write_to_file(q_spectrum, dir + "q_spectrum_old.dat");
    write_to_file(q_old, dir + "q_old.dat");
  }

  workspace.state = boost_klein_gordon_field(param.N, param.L, param.m, tau, workspace.state, 0.01);


  {
    Eigen::VectorXd varphi_old = workspace.state.head(N*N*N);
    Eigen::VectorXd dt_varphi_old = workspace.state.tail(N*N*N);
    write_to_file(varphi_old, dir + "varphi.dat");
    write_to_file(dt_varphi_old, dir + "dt_varphi.dat");
  }

  {
    Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, 0);
    write_to_file(compute_mode_power_spectrum(N, param.L, param.m, 1.0, workspace.state, workspace.fft_wrapper), dir + "varphi_spectrum.dat");
    write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum.dat");
    write_to_file(rho_old, dir + "rho.dat");
  }
  
  {
    Eigen::VectorXd q_old = Equation::compute_momentum_density(workspace, 0);
    const long long int field_size = N*N*N;
    Eigen::VectorXd q_spectrum(3*(N/2)*(N/2)+1);
    q_spectrum.array() = 0;
    for(size_t idx = 0; idx < 3; ++idx){
      Eigen::VectorXd q_idx = q_old.segment(idx * field_size, field_size);
      q_spectrum += compute_power_spectrum(N, q_idx, workspace.fft_wrapper);
    }
    write_to_file(q_spectrum, dir + "q_spectrum.dat");
    write_to_file(q_old, dir + "q.dat");
  }

  // Eigen::VectorXd state_new = boost_klein_gordon_field(param.N, param.L, param.m, tau, workspace.state, 0.01);
  
  // {
  //   Eigen::VectorXd varphi = state_new.head(N*N*N);
  //   Eigen::VectorXd dt_varphi = state_new.tail(N*N*N);
  //   write_to_file(varphi, dir + "varphi.dat");
  //   write_to_file(dt_varphi, dir + "dt_varphi.dat");
  // }
  
  // {
  //   Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, 0);
  //   write_to_file(compute_mode_power_spectrum(N, param.L, param.m, 1.0, workspace.state, workspace.fft_wrapper), dir + "varphi_spectrum.dat");
  //   write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum.dat");
  //   write_to_file(rho_old, dir + "rho.dat");
  // }

  // Eigen::VectorXd varphi;
  // Eigen::VectorXd dt_varphi;
  // {
  //   Workspace workspace(param, unperturbed_grf);
  //   long long int field_size = workspace.state.size() / 2;
  //   varphi = workspace.state.head(field_size);
  //   dt_varphi = workspace.state.tail(field_size);
  // }
  
  // Eigen::VectorXd varphi(N*N*N);
  // Eigen::VectorXd dt_varphi(N*N*N);

  // for(int a = 0; a < N; ++a){
  //   for(int b = 0; b < N; ++b){
  //     for(int c = 0; c < N; ++c){
  // 	varphi(IDX_OF(N, a, b, c)) = cos(2 * std::numbers::pi * c / N);
  // 	dt_varphi(IDX_OF(N, a, b, c)) = 1;
  //     }
  //   }
  // }

}

void generate_ic_proca(void)
{
  using namespace std::numbers;
  // Set the PRNG seed.
  RandomNormal::set_generator_seed(0);

  
  // Set the directory for output.
  const std::string dir = "output/proca_IC/";
  // const std::string dir = "/media/hypermania/Drive_001/FreeStreamingULDM/proca_IC/";
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
  
  // Eigen::VectorXd tau(N*N*N);
  // for(int a = 0; a < N; ++a){
  //   for(int b = 0; b < N; ++b){
  //     for(int c = 0; c < N; ++c){
  // 	tau(IDX_OF(N, a, b, c)) = -0.5 * cos(2 * std::numbers::pi * c / N);
  //     }
  //   }
  // }



  // auto proca_initializer = [](const auto param, auto &workspace) {
  //   const long long int N = param.N;
  //   const long long int lattice_size = N*N*N;
  //   workspace.state = Eigen::VectorXd::Zero(6 * lattice_size);
  // };
  
  Workspace workspace(param, unperturbed_proca_grf);

  Eigen::VectorXd tau;
  {
    Spectrum P_delta_dot = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
    Eigen::VectorXd delta_dot = generate_gaussian_random_field(param.N, param.L, P_delta_dot);
    tau = compute_inverse_laplacian(param.N, param.L, delta_dot, workspace.fft_wrapper);
    std::cout << "max tau = " << tau.maxCoeff() << std::endl;
    std::cout << "min tau = " << tau.minCoeff() << std::endl;
  }
  write_to_file(tau, dir + "tau.dat");
  
  {
    ProcaTransverseProjector projector(N);
    Eigen::VectorXd A = workspace.state.segment(0, 3*N*N*N);
    Eigen::VectorXd dt_A = workspace.state.segment(3*N*N*N, 3*N*N*N);
    projector.proca_project_to_transverse(A, workspace.fft_wrapper);
    projector.proca_project_to_transverse(dt_A, workspace.fft_wrapper);
    workspace.state.segment(0, 3*N*N*N) = A;
    workspace.state.segment(3*N*N*N, 3*N*N*N) = dt_A;
  }
  
  {
    Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, 0);
    write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum_old.dat");
    write_to_file(rho_old, dir + "rho_old.dat");
  }

  {
    const long long int lattice_size = N*N*N;
    Eigen::VectorXd state_x(2*lattice_size);
    Eigen::VectorXd state_y(2*lattice_size);
    Eigen::VectorXd state_z(2*lattice_size);
    state_x.segment(0, lattice_size) = workspace.state.segment(0, lattice_size);
    state_x.segment(lattice_size, lattice_size) = workspace.state.segment(3*lattice_size, lattice_size);
    state_y.segment(0, lattice_size) = workspace.state.segment(lattice_size, lattice_size);
    state_y.segment(lattice_size, lattice_size) = workspace.state.segment(4*lattice_size, lattice_size);
    state_z.segment(0, lattice_size) = workspace.state.segment(2*lattice_size, lattice_size);
    state_z.segment(lattice_size, lattice_size) = workspace.state.segment(5*lattice_size, lattice_size);
    Eigen::VectorXd spectrum = compute_mode_power_spectrum(N, param.L, param.m, 1.0, state_x, workspace.fft_wrapper);
    spectrum += compute_mode_power_spectrum(N, param.L, param.m, 1.0, state_y, workspace.fft_wrapper);
    spectrum += compute_mode_power_spectrum(N, param.L, param.m, 1.0, state_z, workspace.fft_wrapper);
    write_to_file(spectrum, dir + "varphi_spectrum_old.dat");
  }
  
  {
    Eigen::VectorXd q_old = Equation::compute_momentum_density(workspace, 0);
    const long long int field_size = N*N*N;
    Eigen::VectorXd q_spectrum(3*(N/2)*(N/2)+1);
    q_spectrum.array() = 0;
    for(size_t idx = 0; idx < 3; ++idx){
      Eigen::VectorXd q_idx = q_old.segment(idx * field_size, field_size);
      q_spectrum += compute_power_spectrum(N, q_idx, workspace.fft_wrapper);
    }
    write_to_file(q_spectrum, dir + "q_spectrum_old.dat");
    write_to_file(q_old, dir + "q_old.dat");
  }

  // workspace.state = boost_proca_field(param.N, param.L, param.m, tau, workspace.state, 0.01);
  workspace.state = boost_proca_field(param.N, param.L, param.m, tau, workspace.state, 0.01, dir + "state_scratch.dat");

  {
    Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, 0);
    write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum.dat");
    write_to_file(rho_old, dir + "rho.dat");
  }

  {
    const long long int lattice_size = N*N*N;
    Eigen::VectorXd state_x(2*lattice_size);
    Eigen::VectorXd state_y(2*lattice_size);
    Eigen::VectorXd state_z(2*lattice_size);
    state_x.segment(0, lattice_size) = workspace.state.segment(0, lattice_size);
    state_x.segment(lattice_size, lattice_size) = workspace.state.segment(3*lattice_size, lattice_size);
    state_y.segment(0, lattice_size) = workspace.state.segment(lattice_size, lattice_size);
    state_y.segment(lattice_size, lattice_size) = workspace.state.segment(4*lattice_size, lattice_size);
    state_z.segment(0, lattice_size) = workspace.state.segment(2*lattice_size, lattice_size);
    state_z.segment(lattice_size, lattice_size) = workspace.state.segment(5*lattice_size, lattice_size);
    Eigen::VectorXd spectrum = compute_mode_power_spectrum(N, param.L, param.m, 1.0, state_x, workspace.fft_wrapper);
    spectrum += compute_mode_power_spectrum(N, param.L, param.m, 1.0, state_y, workspace.fft_wrapper);
    spectrum += compute_mode_power_spectrum(N, param.L, param.m, 1.0, state_z, workspace.fft_wrapper);
    write_to_file(spectrum, dir + "varphi_spectrum.dat");
  }
  
  {
    Eigen::VectorXd q_old = Equation::compute_momentum_density(workspace, 0);
    const long long int field_size = N*N*N;
    Eigen::VectorXd q_spectrum(3*(N/2)*(N/2)+1);
    q_spectrum.array() = 0;
    for(size_t idx = 0; idx < 3; ++idx){
      Eigen::VectorXd q_idx = q_old.segment(idx * field_size, field_size);
      q_spectrum += compute_power_spectrum(N, q_idx, workspace.fft_wrapper);
    }
    write_to_file(q_spectrum, dir + "q_spectrum.dat");
    write_to_file(q_old, dir + "q.dat");
  }

  // auto At = Equation::compute_At(workspace, 0);
  // std::cout << "before : " << At.squaredNorm() / pow(param.N, 3) << std::endl;

  // Eigen::VectorXd At_predicted(N*N*N);
  // for(int a = 0; a < N; ++a){
  //   for(int b = 0; b < N; ++b){
  //     for(int c = 0; c < N; ++c){
  // 	workspace.state(5*N*N*N + IDX_OF(N, a, b, c)) = cos(2 * std::numbers::pi * c / N);

  // 	double k_IR = 2 * pi / param.L;
  // 	At_predicted(IDX_OF(N, a, b, c)) = sin(2 * std::numbers::pi * c / N) * k_IR / (k_IR * k_IR + param.m * param.m);
  //     }
  //   }
  // }
  
  // At = Equation::compute_At(workspace, 0);
  // std::cout << "after : " << At.squaredNorm() / pow(param.N, 3) << std::endl;
  // std::cout << "difference : " << (At - At_predicted).squaredNorm() / pow(param.N, 3) << std::endl;
  // return;

  // Spectrum P = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);
  // Eigen::VectorXd random_proca = generate_gaussian_random_proca_field(param.N, param.L, P);
  
  /*
  Eigen::VectorXd random_proca(3*N*N*N);
  for(int a = 0; a < N; ++a){
    for(int b = 0; b < N; ++b){
      for(int c = 0; c < N; ++c){
	random_proca(1*N*N*N + IDX_OF(N, a, b, c)) = - cos(2 * std::numbers::pi * c / N);
      }
    }
  }
  // random_proca.head(N*N*N) = 
  ProcaTransverseProjector projector(N);

  std::cout << "before : " << random_proca.squaredNorm() / pow(param.N, 3) << std::endl;
  projector.proca_project_to_transverse(random_proca, workspace.fft_wrapper);
  std::cout << "after : " << random_proca.squaredNorm() / pow(param.N, 3) << std::endl;
  projector.proca_project_to_transverse(random_proca, workspace.fft_wrapper);
  std::cout << "check idempotence : " << random_proca.squaredNorm() / pow(param.N, 3) << std::endl;

  return;
  */
  
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

void generate_ic_sp(void)
{
  using namespace std::numbers;
  // Set the PRNG seed.
  RandomNormal::set_generator_seed(0);

  
  // Set the directory for output.
  // const std::string dir = "output/SP_infalling_IC/";
  // const std::string dir = "output/SP_IC/";
  const std::string dir = "/media/hypermania/Drive_001/FreeStreamingULDM/SP_IC/";
  // const std::string dir = "/media/hypermania/Drive_001/FreeStreamingULDM/SP_infalling_IC/";
  prepare_directory_for_output(dir);

  
  // Set parameters for the simulation.
  // We use units in which a_eq = 1, H_eq = 1.
  // MyParam param
  //   {
  //     .N = 384, // Lattice points per axis
  //     .L = 25.132741228718345908, // Size of the box
  //     // ULDM params
  //     .m = 1e1, // Mass of scalar field
  //     .lambda = 0, // Lambda phi^4 coupling strength
  //     //.f_a = 30.0, // Not relevant for ComovingCurvatureEquationInFRW
  //     .k_ast = 16.0, // Characteristic momentum
  //     .k_Psi = 1.0, // Not relevant for ComovingCurvatureEquationInFRW
  //     .varphi_std_dev = 1.0, // Standard deviation of field
  //     .Psi_std_dev = 0.1, // Standard deviation of metric perturbation Psi
  //     // FRW metric params
  //     .a1 = 64.0,
  //     .H1 = 1.0 / 512.0,
  //     .t1 = 2.0 / (3 * param.H1),
  //     // Start and end time for numerical integration, and time interval between saves
  //     .t_start = param.t1,
  //     .t_end = param.t_start + (pow(3.5 / param.a1, 2) - 1.0) / (2 * param.H1),
  //     .t_interval = 49.99, // Save a snapshot every t_interval
  //     // Numerical method parameter
  //     .delta_t = 0.5, // Time step for numerical integration
  //     // Psi approximation parameter
  //     .M = 128 // Lattice points for storing / computing Psi
  //   };

  // Set parameters for the simulation.
  // We use units in which a_eq = 1, H_eq = 1.
  MyParam param
    {
      .N = 384,
      .L = 12.566370614359172954,
      // ULDM params
      .m = 10.000000000000000000,
      .lambda = 0,
      //.f_a = 30.0,
      .k_ast = 24.000000000000000000,
      .k_Psi = 1.0,
      .varphi_std_dev = 1.0,
      .Psi_std_dev = 0.1,
      // FRW metric params
      .a1 = 16.000000000000000000,
      .H1 = pow(param.a1, -1.5),
      .t1 = 2.0 / (3 * param.H1),
      // Start and end time for numerical integration, and time interval between saves
      .t_start = param.t1,
      .t_end = param.t_start + (pow(3.5 / param.a1, 2) - 1.0) / (2 * param.H1),
      .t_interval = 49.99, // Save a snapshot every t_interval
      // Numerical method parameter
      .delta_t = 0.5, // Time step for numerical integration
      // Psi approximation parameter
      .M = 128
    };
  print_param(param);
  save_param_for_Mathematica(param, dir);

  
  typedef SchrodingerPoissonEquation Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;

  const long long int N = param.N;
  
  Workspace workspace(param, matter_dominated_sp_grf);
  // Workspace workspace(param, infalling_sp_grf);
  
  {
    const long long int lattice_size = N*N*N;
    Eigen::VectorXd psi_1_re(lattice_size);
    Eigen::VectorXd psi_1_im(lattice_size);
    Eigen::VectorXd psi_2_re(lattice_size);
    Eigen::VectorXd psi_2_im(lattice_size);
    Eigen::VectorXd psi_3_re(lattice_size);
    Eigen::VectorXd psi_3_im(lattice_size);
    psi_1_re = workspace.state.segment(0, lattice_size).real();
    psi_1_im = workspace.state.segment(0, lattice_size).imag();
    psi_2_re = workspace.state.segment(0, lattice_size).real();
    psi_2_im = workspace.state.segment(0, lattice_size).imag();
    psi_3_re = workspace.state.segment(0, lattice_size).real();
    psi_3_im = workspace.state.segment(0, lattice_size).imag();

    Eigen::VectorXd spectrum = compute_power_spectrum(N, psi_1_re, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_1_im, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_2_re, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_2_im, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_3_re, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_3_im, workspace.fft_wrapper);
    write_to_file(spectrum, dir + "varphi_spectrum_old.dat");
  }
  {
    Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, param.t_start);
    write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum_old.dat");
    write_to_file(rho_old, dir + "rho_old.dat");
  }
  {
    Eigen::VectorXd q_old = Equation::compute_momentum_density(workspace, param.t_start);
    const long long int field_size = N*N*N;
    Eigen::VectorXd q_spectrum(3*(N/2)*(N/2)+1);
    q_spectrum.array() = 0;
    for(size_t idx = 0; idx < 3; ++idx){
      Eigen::VectorXd q_idx = q_old.segment(idx * field_size, field_size);
      q_spectrum += compute_power_spectrum(N, q_idx, workspace.fft_wrapper);
    }
    write_to_file(q_spectrum, dir + "q_spectrum_old.dat");
    write_to_file(q_old, dir + "q_old.dat");
  }

  
  // Eigen::VectorXd tau(N*N*N);
  // for(int a = 0; a < N; ++a){
  //   for(int b = 0; b < N; ++b){
  //     for(int c = 0; c < N; ++c){
  // 	tau(IDX_OF(N, a, b, c)) = -0.5 * cos(2 * 2 * std::numbers::pi * c / N) * 0.05;
  //     }
  //   }
  // }
  // workspace.state = boost_sp_field(param.N, param.L, param.m, tau, workspace.state);
  
  workspace.state = boost_sp_field(param.N, param.L, param.m, workspace.tau, workspace.state);
  
  {
    const long long int lattice_size = N*N*N;
    Eigen::VectorXd psi_1_re(lattice_size);
    Eigen::VectorXd psi_1_im(lattice_size);
    Eigen::VectorXd psi_2_re(lattice_size);
    Eigen::VectorXd psi_2_im(lattice_size);
    Eigen::VectorXd psi_3_re(lattice_size);
    Eigen::VectorXd psi_3_im(lattice_size);
    psi_1_re = workspace.state.segment(0, lattice_size).real();
    psi_1_im = workspace.state.segment(0, lattice_size).imag();
    psi_2_re = workspace.state.segment(0, lattice_size).real();
    psi_2_im = workspace.state.segment(0, lattice_size).imag();
    psi_3_re = workspace.state.segment(0, lattice_size).real();
    psi_3_im = workspace.state.segment(0, lattice_size).imag();

    Eigen::VectorXd spectrum = compute_power_spectrum(N, psi_1_re, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_1_im, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_2_re, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_2_im, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_3_re, workspace.fft_wrapper);
    spectrum += compute_power_spectrum(N, psi_3_im, workspace.fft_wrapper);
    write_to_file(spectrum, dir + "varphi_spectrum.dat");
  }
  {
    Eigen::VectorXd rho_old = Equation::compute_energy_density(workspace, param.t_start);
    write_to_file(compute_power_spectrum(N, rho_old, workspace.fft_wrapper), dir + "rho_spectrum.dat");
    write_to_file(rho_old, dir + "rho.dat");
  }
  {
    Eigen::VectorXd q_old = Equation::compute_momentum_density(workspace, param.t_start);
    const long long int field_size = N*N*N;
    Eigen::VectorXd q_spectrum(3*(N/2)*(N/2)+1);
    q_spectrum.array() = 0;
    for(size_t idx = 0; idx < 3; ++idx){
      Eigen::VectorXd q_idx = q_old.segment(idx * field_size, field_size);
      q_spectrum += compute_power_spectrum(N, q_idx, workspace.fft_wrapper);
    }
    write_to_file(q_spectrum, dir + "q_spectrum.dat");
    write_to_file(q_old, dir + "q.dat");
  }

  
  // Eigen::VectorXd tau;
  // {
  //   Spectrum P_delta_dot = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.Psi_std_dev, param.k_Psi, -3);
  //   Eigen::VectorXd delta_dot = generate_gaussian_random_field(param.N, param.L, P_delta_dot);
  //   tau = compute_inverse_laplacian(param.N, param.L, delta_dot, workspace.fft_wrapper);
  //   std::cout << "max tau = " << tau.maxCoeff() << std::endl;
  //   std::cout << "min tau = " << tau.minCoeff() << std::endl;
  // }
  // write_to_file(tau, dir + "tau.dat");

}

void solve_field_equation(void)
{
  using namespace Eigen;
  using namespace boost::numeric::odeint;

  
  // Set the PRNG seed.
  RandomNormal::set_generator_seed(0);

  
  // Set the directory for output.
  const std::string dir = "output/Growth_and_FS/";
  prepare_directory_for_output(dir);

  
  // Set parameters for the simulation.
  MyParam param
    {
     .N = 384, // Lattice points per axis
     .L = 384 * 0.8, // Size of the box
     // ULDM params
     .m = 1.0, // Mass of scalar field
     .lambda = 0, // Lambda phi^4 coupling strength
     //.f_a = 30.0, // Not relevant for ComovingCurvatureEquationInFRW
     .k_ast = 1.0, // Characteristic momentum
     .k_Psi = 1.0, // Not relevant for ComovingCurvatureEquationInFRW
     .varphi_std_dev = 1.0, // Standard deviation of field
     .Psi_std_dev = 0.02, // Standard deviation of metric perturbation Psi
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

  
  // Choose an equation to solve.
  // Here we solve a scalar field equation with background metric perturbations.
  // Also see CudaApproximateComovingCurvatureEquationInFRW, which is a CUDA implementation of the same equation.
  typedef ComovingCurvatureEquationInFRW Equation;
  //typedef CudaApproximateComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;

  
  // Initialize the workspace given params and a procedure for setting initial conditions.
  // The initialization procedure is described in Sec.3 of the paper.
  Workspace workspace(param, perturbed_grf_and_comoving_curvature_fft);

  
  // The equation object.
  Equation eqn(workspace);

  
  // Choose what to save in the course of simulation.
  // Here we save the field spectrum, density spectrum, and 2D density slices.
  ConstIntervalObserver<Equation, true, true, true> observer(dir, param, eqn);

  
  // Choose the numerical integrator.
  // Here we use RK4, you can also use other methods.
  // See https://www.boost.org/doc/libs/1_85_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/getting_started/overview.html .
  auto stepper = runge_kutta4_classic<State, double, State, double>();
  // auto stepper = make_controlled(1e-9, 1e-9, runge_kutta_fehlberg78<State, double, State, double>());

  
  {
    // Save spectrum for R and initial potential Psi
    double eta_i = workspace.cosmology.eta(param.t_start);
    auto kernel = [eta_i](double k){
		    return k == 0.0 ? 0.0 : (6 * sqrt(3) * (-((k * eta_i * cos((k * eta_i) / sqrt(3))) / sqrt(3)) + sin((k * eta_i) / sqrt(3)))) / (pow(k, 3) * pow(eta_i, 3));
		  };
    
    Eigen::VectorXd R_fft_eigen(workspace.R_fft.size());
    copy_vector(R_fft_eigen, workspace.R_fft);
    
    auto fft_wrapper = fftwWrapper(param.N);
    Eigen::VectorXd R = fft_wrapper.execute_z2d(R_fft_eigen) / pow(param.N, 3);
    Eigen::VectorXd Psi = compute_field_with_scaled_fourier_modes(param.N, param.L, R, kernel, fft_wrapper);

    std::cout << "Psi_std_dev = " << sqrt(Psi.squaredNorm() / pow(param.N, 3)) << '\n';
    auto Psi_spectrum = compute_power_spectrum(param.N, Psi, fft_wrapper);
    write_to_file(Psi_spectrum, dir + "initial_Psi_spectrum.dat");
    
    auto R_spectrum = compute_power_spectrum(param.N, R, fft_wrapper);
    write_to_file(R_spectrum, dir + "initial_R_spectrum.dat");
  }

  
  // Solve the equation.
  run_and_measure_time("Solving equation",
  		       [&](){
			 int num_steps = integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, param.delta_t, observer);
			 std::cout << "total number of steps = " << num_steps << '\n';
		       } );
  
  write_vector_to_file(workspace.t_list, dir + "t_list.dat");
  
  
  // Optional: save the final state.
  {
    Eigen::VectorXd state_out(workspace.state.size());
    copy_vector(state_out, workspace.state);
    write_to_file(state_out, dir + "state.dat");
  }
}


void generate_wkb_solutions(void)
{
  using namespace Eigen;
  using namespace boost::numeric::odeint;
  
  // Set the seed for PRNG for consistency
  RandomNormal::set_generator_seed(0);
  
  // Set directory for output
  const std::string dir = "output/Growth_and_FS/";
  //prepare_directory_for_output(dir);

  // Set parameters for the simulation
  MyParam param
    {
     .N = 384,
     .L = 384 * 0.8,
     // ULDM params
     .m = 1.0,
     .lambda = 0,
     .k_ast = 1.0,
     .k_Psi = 1.0,
     .varphi_std_dev = 1.0,
     .Psi_std_dev = 0.02,
     // FRW metric params
     .a1 = 1.0,
     .H1 = 0.05,
     .t1 = 1.0 / (2 * param.H1),
     // Solution record params
     .t_start = param.t1,
     .t_end = param.t_start + (pow(60.0 / param.a1, 2) - 1.0) / (2 * param.H1),
     .t_interval = 49.99,
     // Numerical method parameter
     .delta_t = 0.1,
     // Psi approximation parameter
     .M = 128
    };
  //print_param(param);
  //save_param_for_Mathematica(param, dir);

  // Setup a workspace
  typedef KleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  Workspace workspace(param, unperturbed_grf);

  // Load final state from directory
  workspace.state = load_VectorXd_from_file(dir + "state.dat");

  // Prepare WKB solution class
  WKBSolutionForKleinGordonEquationInFRW wkb(workspace, param.t_end);

  // Specify the time to evaluate the WKB solution
  Eigen::VectorXd times = param.t_end * Eigen::VectorXd::LinSpaced(180, 0.09, 16.2).array().exp();

  // Evaluate the WKB solutions and save them
  for(long int i = 0; i < times.size(); ++i) {
    const int N = param.N;
    double t_eval = times[i];
    workspace.state = wkb.evaluate_at(t_eval);
    {
      auto rho = Equation::compute_energy_density(workspace, t_eval);
      auto rho_spectrum = compute_power_spectrum(param.N, rho, workspace.fft_wrapper);
      write_to_filename_template(rho_spectrum, dir + "wkb_rho_spectrum_%d.dat", i);
      
      Eigen::VectorXd rho_slice = rho.head(N*N); // The density for a = 0 slice.
      Eigen::VectorXd rho_axis_average = rho.reshaped(N*N, N).rowwise().mean(); // The density overaged over a axis.
      write_to_filename_template(rho_slice, dir + "wkb_rho_slice_%d.dat", i);
      write_to_filename_template(rho_axis_average, dir + "wkb_rho_axis_average_%d.dat", i);
    }
    {
      auto varphi_plus_spectrum = compute_mode_power_spectrum(N, param.L, param.m, workspace.cosmology.a(t_eval), workspace.state, workspace.fft_wrapper);
      write_to_filename_template(varphi_plus_spectrum, dir + "wkb_varphi_plus_spectrum_%d.dat", i);
    }
  }

  write_to_file(times, dir + "wkb_t_list.dat");

}


