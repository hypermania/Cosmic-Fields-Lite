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


int main(int argc, char **argv){
  // Runs the simulation described in Section 4.2.2 of paper.
  
  // Solve scalar field equation in a background of comoving curvature perturbation.
  // Save output to output/Growth_and_FS/
  solve_field_equation();

  // Use WKB solution to extend the simulation.
  generate_wkb_solutions();
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
     // Solution record params
     .t_start = param.t1,
     .t_end = param.t_start + (pow(60.0 / param.a1, 2) - 1.0) / (2 * param.H1),
     .t_interval = 49.99, // Save a snapshot every t_interval
     // Numerical method parameter
     .delta_t = 0.1, // Time step for numerical integration
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
    write_VectorXd_to_file(Psi_spectrum, dir + "initial_Psi_spectrum.dat");
    
    auto R_spectrum = compute_power_spectrum(param.N, R, fft_wrapper);
    write_VectorXd_to_file(R_spectrum, dir + "initial_R_spectrum.dat");
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
    write_VectorXd_to_file(state_out, dir + "state.dat");
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
      write_VectorXd_to_filename_template(rho_spectrum, dir + "wkb_rho_spectrum_%d.dat", i);
      
      Eigen::VectorXd rho_slice = rho.head(N*N); // The density for a = 0 slice.
      Eigen::VectorXd rho_axis_average = rho.reshaped(N*N, N).rowwise().mean(); // The density overaged over a axis.
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


