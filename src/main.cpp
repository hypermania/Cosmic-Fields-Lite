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

#include "param.hpp"
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

#ifndef DISABLE_CUDA
#include <thrust/device_vector.h>
#include "equations_cuda.cuh"
#include "cufft.h"
#include "fdm3d_cuda.cuh"
#endif

#include "tests.hpp"

#define GRID_SIZE 384

void run_eqn_with_lambda(const std::string dir, double lambda, double varphi_std_dev);

// The struct containing parameters for the simulation.
// Numbers within are fiducial values.
struct MyParam {
  // lattice params
  long long int N = GRID_SIZE;
  double L = GRID_SIZE;
  // ULDM params
  double m = 1.0;
  double lambda = 0;
  double f_a = 10.0;
  double k_ast = 0.5;
  double k_Psi = 0.1;
  double varphi_std_dev = 0.1;
  double Psi_std_dev = 0.15;
  // FRW metric params
  double a1 = 1.0;
  double H1 = 0.1;
  double t1 = 1.0 / (2 * 0.1);
  // Solution record params
  double t_start = 1.0 / (2 * 0.1);
  double t_end = 1.0 / (2 * 0.1) + (pow(3.0 / 1.0, 2) - 1.0) / (2 * 0.1);
  double t_interval = 0.1;
  // Numerical method parameter
  double delta_t = 0.1;
  // Psi approximation parameter (the size of the grid storing Psi)
  long long int M = GRID_SIZE / 3;
  // Params for adding fluctuations on a homogeneous background
  double f = 30;
  double delta_varphi_std_dev = 0.001;
  double k_delta_varphi = 1.0;
};

void test_wkb()
{
  using namespace Eigen;
  using namespace boost::numeric::odeint;
  
  // Set the seed for PRNG for consistency
  RandomNormal::set_generator_seed(0);
  
  // Set directory for output
  //const std::string dir = "output/Growth_and_FS_2_small_time_step/";
  const std::string dir = "output/FS_Without_Gravity/";
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
     .k_Psi = 1.0, //1.0 / 6.0,
     .varphi_std_dev = 1.0,
     .Psi_std_dev = 0.02,
     // FRW metric params
     .a1 = 1.0,
     .H1 = 0.05,
     .t1 = 1.0 / (2 * param.H1),
     // Solution record params
     .t_start = param.t1,
     .t_end = param.t_start + (pow(60.0 / param.a1, 2) - 1.0) / (2 * param.H1),
     .t_interval = 49.99, //(param.t_end - param.t_start) / 2.0,
     // Numerical method parameter
     .delta_t = 0.1,
     // Psi approximation parameter
     .M = 128
    };
  //print_param(param);
  //save_param_for_Mathematica(param, dir);

  // Decide which equation to solve
  //typedef ComovingCurvatureEquationInFRW Equation;
  typedef KleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  
  // Initialize the workspace with the params (1st argument) and an initilizer procedure (2nd argument)
  Workspace workspace(param, unperturbed_grf);
  //Workspace workspace(param, wave_packet);
  
  Equation eqn(workspace);
  
  workspace.state = load_VectorXd_from_file(dir + "state.dat");

  
  WKBSolutionForKleinGordonEquationInFRW wkb(workspace, param.t_end);

  Eigen::VectorXd times = param.t_end * Eigen::VectorXd::LinSpaced(180, 0.09, 16.2).array().exp();

  
  for(size_t i = 0; i < times.size(); ++i) {
    const int N = param.N;
    double t_eval = times[i];
    workspace.state = wkb.evaluate_at(t_eval);
    {
      auto rho = Equation::compute_energy_density(workspace, t_eval);
      auto rho_spectrum = compute_power_spectrum(param.N, rho, workspace.fft_wrapper);
      write_VectorXd_to_filename_template(rho_spectrum, dir + "wkb_rho_spectrum_%d.dat", i);
      
      Eigen::VectorXd rho_slice = rho.head(N*N);
      Eigen::VectorXd rho_axis_average = rho.reshaped(N*N, N).rowwise().mean();
      write_VectorXd_to_filename_template(rho_slice, dir + "wkb_rho_slice_%d.dat", i);
      write_VectorXd_to_filename_template(rho_axis_average, dir + "wkb_rho_axis_average_%d.dat", i);
    }
    {
      auto varphi_plus_spectrum = compute_mode_power_spectrum(N, param.L, param.m, workspace.state, workspace.fft_wrapper);
      write_VectorXd_to_filename_template(varphi_plus_spectrum, dir + "wkb_varphi_plus_spectrum_%d.dat", i);
    }
  }

  write_VectorXd_to_file(times, dir + "wkb_t_list.dat");

}


void soliton_formation(void){
  using namespace Eigen;
  using namespace boost::numeric::odeint;
  
  // Set the seed for PRNG for consistency
  RandomNormal::set_generator_seed(0);
  
  // Set directory for output
  const std::string dir = "output/Soliton_3/";
  //prepare_directory_for_output(dir);

  // Set parameters for the simulation
  MyParam param
    {
     .N = 384,
     .L = 384 * 2.0,
     // ULDM params
     .m = 1.0,
     .lambda = 0,
     .f_a = 30.0,
     .k_ast = 0.1,
     .k_Psi = 0.03,
     .varphi_std_dev = 1.0,
     .Psi_std_dev = 0.2,
     // FRW metric params
     .a1 = 1.0,
     .H1 = 0.0,
     .t1 = 0.0,
     // Solution record params
     .t_start = param.t1,
     .t_end = 12500,
     .t_interval = 49.99,
     // Numerical method parameter
     .delta_t = 0.05,
     // Psi approximation parameter
     .M = 128,
     // fluctuation parameters
     .f = param.f_a * 2,
     .delta_varphi_std_dev = 0.01,
     .k_delta_varphi = 0.125 * (param.N / 2) / param.L
    };
  print_param(param);
  //save_param_for_Mathematica(param, dir);

  // Decide which equation to solve
  //typedef CudaKleinGordonEquationInFRW Equation;
  //typedef CudaComovingCurvatureEquationInFRW Equation;
  //typedef CudaApproximateComovingCurvatureEquationInFRW Equation;
  typedef CudaSqrtPotentialEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  
  //Workspace workspace(param, homogeneous_field_with_fluctuations);
  Workspace workspace(param, perturbed_grf);

  {
    auto Psi_spectrum = compute_power_spectrum(param.N, workspace.Psi, workspace.fft_wrapper);
    Eigen::VectorXd Psi_spectrum_eigen(Psi_spectrum.size());
    copy_vector(Psi_spectrum_eigen, Psi_spectrum);
    write_VectorXd_to_file(Psi_spectrum_eigen, dir + "initial_Psi_spectrum.dat");
  }

  exit(0);
  
  Equation eqn(workspace);
  const_interval_observer<Equation, true, true, true> observer(dir, param, eqn);

  // Solve the equation
  auto stepper = runge_kutta4_classic<State, double, State, double>();
  
  run_and_measure_time("Solving equation",
  		       [&](){
			 int num_steps = integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, param.delta_t, observer);
			 std::cout << "total number of steps = " << num_steps << '\n';
		       } );
  
  write_vector_to_file(workspace.t_list, dir + "t_list.dat");
  
  
  // Optional: save the final state
  {
    Eigen::VectorXd state_out(workspace.state.size());
    copy_vector(state_out, workspace.state);
    write_VectorXd_to_file(state_out, dir + "state.dat");
  }
}  


int main(int argc, char **argv){
  //soliton_formation();
  //test_03();
  //exit(0);
  
  using namespace Eigen;
  using namespace boost::numeric::odeint;
  
  // Set the seed for PRNG for consistency
  RandomNormal::set_generator_seed(0);
  
  // Set directory for output
  //const std::string dir = "output/Soliton_1/";
  //const std::string dir = "output/Growth_and_FS_2_small_time_step/";
  //const std::string dir = "output/FS_Without_Gravity/";
  //const std::string dir = "output/Growth_and_FS_2_256/";
  const std::string dir = "output/Growth_and_FS_2_256_fix_dx/";
  prepare_directory_for_output(dir);

  // Set parameters for the simulation
  MyParam param
    {
     .N = 256,
     .L = 256 * 0.8, // 384 * 2.0,
     // ULDM params
     .m = 1.0,
     .lambda = 0,
     //.f_a = 30.0,
     .k_ast = 1.0,
     .k_Psi = 1.0, //1.0 / 6.0,
     .varphi_std_dev = 1.0,
     .Psi_std_dev = 0.02,
     // FRW metric params
     .a1 = 1.0,
     .H1 = 0.05,
     .t1 = 1.0 / (2 * param.H1),
     // Solution record params
     .t_start = param.t1,
     .t_end = param.t_start + (pow(60.0 / param.a1, 2) - 1.0) / (2 * param.H1),
     .t_interval = 49.99, //(param.t_end - param.t_start) / 2.0,
     // Numerical method parameter
     .delta_t = 0.1,
     // Psi approximation parameter
     .M = 128
    };
  print_param(param);
  save_param_for_Mathematica(param, dir);

  // Decide which equation to solve
  //typedef CudaKleinGordonEquationInFRW Equation;
  //typedef CudaComovingCurvatureEquationInFRW Equation;
  typedef CudaApproximateComovingCurvatureEquationInFRW Equation;
  //typedef CudaSqrtPotentialEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  
  // Initialize the workspace with the params (1st argument) and an initilizer procedure (2nd argument)
  //Workspace workspace(param, perturbed_grf);
  Workspace workspace(param, perturbed_grf_and_comoving_curvature_fft);
  
  // {
  //   decltype(workspace.R_fft) temp_vec;
  //   workspace.R_fft.swap(temp_vec);
  // }
  
  // {
  //   auto Psi_spectrum = compute_power_spectrum(param.N, workspace.Psi, workspace.fft_wrapper);
  //   Eigen::VectorXd Psi_spectrum_eigen(Psi_spectrum.size());
  //   copy_vector(Psi_spectrum_eigen, Psi_spectrum);
  //   write_VectorXd_to_file(Psi_spectrum_eigen, dir + "initial_Psi_spectrum.dat");
  // }
  
  
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
    write_VectorXd_to_file(Psi_spectrum, dir + "initial_Psi_spectrum.dat");
    
    auto R_spectrum = compute_power_spectrum(param.N, R, fft_wrapper);
    write_VectorXd_to_file(R_spectrum, dir + "initial_R_spectrum.dat");
  }
  
  Equation eqn(workspace);
  //const_interval_observer observer(dir, param, eqn);
  const_interval_observer<Equation, true, true, true> observer(dir, param, eqn);

  // Solve the equation
  auto stepper = runge_kutta4_classic<State, double, State, double>();
  //auto stepper = midpoint<State, double, State, double>();
  //auto stepper = modified_midpoint<State, double, State, double>();
  //auto stepper = adams_bashforth_moulton<5, State>();
  //stepper.initialize(eqn, workspace.state, param.t_start, 0.5);
  
  run_and_measure_time("Solving equation",
  		       [&](){
			 int num_steps = integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, param.delta_t, observer);
			 std::cout << "total number of steps = " << num_steps << '\n';
		       } );
  
  write_vector_to_file(workspace.t_list, dir + "t_list.dat");
  
  /*
  // Old code for using an adaptive 8-th order Runge-Kutta algorithm
  auto stepper = make_controlled(1e-9, 1e-9, runge_kutta_fehlberg78<State, double, State, double>());
  */
  
  
  // Optional: save the final state
  {
    Eigen::VectorXd state_out(workspace.state.size());
    copy_vector(state_out, workspace.state);
    write_VectorXd_to_file(state_out, dir + "state.dat");
  }
}  

/*
void run_eqn_with_lambda(const std::string dir, double lambda, double varphi_std_dev){
  using namespace boost::numeric::odeint;
  
  RandomNormal::set_generator_seed(0);
  
  prepare_directory_for_output(dir);
  MyParam param
    {
     .N = GRID_SIZE,
     .L = param.N * 100.0,
     // ULDM params
     .m = 1.0,
     .lambda = 0,
     .f_a = 0.0,
     .k_ast = 0.01,
     .k_Psi = 0.003,
     .varphi_std_dev = 1.0,
     .Psi_std_dev = 0.02,
     // // FRW metric params
     // .a1 = 1.0,
     // .H1 = 0,
     // .t1 = 0,
     // // Solution record params
     // .t_start = param.t1,
     // .t_end = 5000,
     // .t_interval = 20,
     // FRW metric params
     .a1 = 1.0,
     .H1 = 0.005,
     .t1 = 1.0 / (2 * param.H1),
     // Solution record params
     .t_start = param.t1,
     .t_end = param.t_start + (pow(9.0 / param.a1, 2) - 1.0) / (2 * param.H1),
     .t_interval = 20,
     // ULDM background params
     .f = 1,
     .dt_f = 0
    };
  param.lambda = lambda;
  param.varphi_std_dev = varphi_std_dev;
  print_param(param);
  save_param_for_Mathematica(param, dir);
  
  //typedef KleinGordonEquation Equation;
  //typedef CudaLambdaEquationInFRW Equation;
  //typedef CudaFixedCurvatureEquationInFRW Equation;
  //typedef CudaSqrtPotentialEquationInFRW Equation;
  typedef CudaComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, perturbed_grf_and_comoving_curvature_fft);
  Equation eqn(workspace);
  //const_interval_observer<Equation, true, true, false, false, true> observer(dir, param, eqn);
  const_interval_observer observer(dir, param, eqn);

  auto stepper = runge_kutta4<State, double, State, double>();
  run_and_measure_time("Solving equation",
  		       [&](){ integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, 0.1 / param.m, observer); } );
  
  write_vector_to_file(workspace.t_list, dir + "t_list.dat");
}
*/
