#include "tests.hpp"

struct MyParam2 {
  // lattice params
  long long int N = 384;
  double L = 384 * 0.4;
  // ULDM params
  double m = 1.0;
  double lambda = 0;
  double f_a = 20.0;
  double k_ast = 0.1;
  double k_Psi = 0.03;
  double varphi_std_dev = 1.0;
  double Psi_std_dev = 0.15;
  // FRW metric params
  double a1 = 1.0;
  double H1 = 0;
  double t1 = 0;
  // Solution record params
  double t_start = 0;
  double t_end = 1000;
  double t_interval = 20;
  // ULDM background params
  double f = 0;
  double dt_f = 0;
};


#ifndef DISABLE_CUDA

struct MyParam {
  long long int N = 256;
  double L = 256 * 10.0;
  double m = 1.0;
  double lambda = 0.01;
  double k_ast = 0.1;
  double k_Psi = 0.03;
  double varphi_std_dev = 1.0;
  double Psi_std_dev = 0.15;
  double a1 = 1;
  double H1 = 0;
  double t1 = 0;
  double t_start = 0;
  double t_end = 100; //100 * 2 * std::numbers::pi;
  double t_interval = 1;
  double f = 0;
  double dt_f = 1;
};

void test_cuda_00(void)
{
  int N = 1 << 20;
  thrust::device_vector<double> thrust_vec(N);
  Eigen::VectorXd eigen_vec(thrust_vec.size());
  copy_vector(eigen_vec, thrust_vec);
  assert(eigen_vec.norm() == 0.0);
}

void test_cuda_01(void)
{
  typedef CudaLambdaEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  Workspace workspace(param, unperturbed_grf);

  Eigen::VectorXd state(workspace.state.size());
  copy_vector(state, workspace.state);
  Eigen::VectorXd spectrum_1 = compute_power_spectrum(workspace.N, state);

  thrust::device_vector<double> spectrum_thrust = compute_power_spectrum(workspace.N, workspace.state, workspace.fft_wrapper); //fft_d2z);
  Eigen::VectorXd spectrum_2(spectrum_thrust.size());
  copy_vector(spectrum_2, spectrum_thrust);

  std::cout << "spectrum_1.norm() = " << spectrum_1.norm() << '\n';
  std::cout << "diff.norm() = " << (spectrum_1 - spectrum_2).norm() << '\n';
  assert((spectrum_1 - spectrum_2).norm() / spectrum_1.norm() < 1e-14);
}

void test_cuda_02(void)
{
  typedef CudaLambdaEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  param.N = 384;
  
  show_gpu_memory_usage();
  size_t workSize;
  cufftEstimate3d(param.N, param.N, param.N, CUFFT_D2Z, &workSize);
  std::cout << "1. workSize = " << workSize << '\n';

  cufftHandle plan;
  cufftCreate(&plan);

  cufftSetAutoAllocation(plan, 0);
  
  cufftGetSize3d(plan, param.N, param.N, param.N, CUFFT_D2Z, &workSize);
  std::cout << "2. workSize = " << workSize << '\n';
  
  cufftMakePlan3d(plan, param.N, param.N, param.N, CUFFT_D2Z, &workSize);
  std::cout << "3. workSize = " << workSize << '\n';

  cufftGetSize(plan, &workSize);
  std::cout << "4. workSize = " << workSize << '\n';
  
  show_gpu_memory_usage();
  
  cufftDestroy(plan);
}

void test_cuda_03(void)
{
  show_gpu_memory_usage();
  
  {
    cufftWrapper wrapper_all(256);
    cufftWrapper wrapper_d2z(256);
  }
  
  show_gpu_memory_usage();
}

void test_cuda_04(void)
{
  typedef CudaLambdaEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  Workspace workspace(param, unperturbed_grf);

  Eigen::VectorXd state(workspace.state.size());
  copy_vector(state, workspace.state);
  
  //Eigen::VectorXd spectrum_1 = compute_power_spectrum(workspace.N, state);
  //auto ffts = workspace.fft_wrapper.execute_batched_d2z(workspace.state);
  
  run_and_measure_time("run kernel", [&](){
				       for(int i = 0; i < 1000; ++i)
					 //compute_mode_power_spectrum(param.N, param.L, param.m, ffts);
					 Equation::compute_energy_density(workspace, 0.0);
				       cudaStreamSynchronize(0);
				     } );
}

void test_cuda_05(void)
{
  typedef CudaLambdaEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  param.N = 384;
  Workspace workspace(param, unperturbed_grf);

  //auto ffts = workspace.fft_wrapper.execute_batched_d2z(workspace.state);
  auto spectrum = compute_mode_power_spectrum(param.N, param.L, param.m, workspace.state, workspace.fft_wrapper);
  Eigen::VectorXd spectrum_eigen_1(spectrum.size());
  copy_vector(spectrum_eigen_1, spectrum);

  Eigen::VectorXd state_eigen(workspace.state.size());
  copy_vector(state_eigen, workspace.state);
  Eigen::VectorXd f = state_eigen.head(param.N*param.N*param.N);
  Eigen::VectorXd dtf = state_eigen.tail(param.N*param.N*param.N);
  auto f_spectrum = compute_power_spectrum(param.N, f);
  auto dtf_spectrum = compute_power_spectrum(param.N, dtf);
  Eigen::VectorXd inverse_omega_k_sqr = (param.m * param.m + pow(2*std::numbers::pi/param.L, 2) * Eigen::ArrayXd::LinSpaced(f_spectrum.size(), 0, f_spectrum.size() - 1)).inverse().matrix();
  Eigen::VectorXd varphi_plus_spectrum = f_spectrum + dtf_spectrum.cwiseProduct(inverse_omega_k_sqr);

  std::cout << "diff.norm() = " << (spectrum_eigen_1 - varphi_plus_spectrum).norm() << '\n';
  std::cout << "norm() = " << varphi_plus_spectrum.norm() << '\n';

  assert((spectrum_eigen_1 - varphi_plus_spectrum).norm() / varphi_plus_spectrum.norm() < 1e-14);
}


void test_cuda_06(void)
{
  typedef CudaLambdaEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  param.N = 512;
  Workspace workspace(param, unperturbed_grf);

  run_and_measure_time("timed section", [&](){
  Eigen::VectorXd state_eigen(workspace.state.size());
  copy_vector(state_eigen, workspace.state);
  Eigen::VectorXd f = state_eigen.head(param.N*param.N*param.N);
  
  auto fft = workspace.fft_wrapper.execute_d2z(workspace.state);
  auto inv_fft = workspace.fft_wrapper.execute_z2d(fft);
  Eigen::VectorXd inv_fft_eigen(inv_fft.size());
  copy_vector(inv_fft_eigen, inv_fft);
  inv_fft_eigen /= param.N * param.N * param.N;
  
  std::cout << "diff.norm() = " << (inv_fft_eigen - f).norm() << '\n';
  std::cout << "f.norm() = " << f.norm() << '\n';
  std::cout << "inv_fft_eigen.norm() = " << inv_fft_eigen.norm() << '\n';

  assert((inv_fft_eigen - f).norm() / f.norm() < 1e-14);
					} );
}


void test_cuda_07(void)
{
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  param.N = 384;
  param.L = 384.0;
  param.k_ast = 0.1;
  Workspace workspace(param, unperturbed_grf);

  auto rho = Equation::compute_energy_density(workspace, 0);
  auto Phi = compute_inverse_laplacian(param.N, param.L, rho, workspace.fft_wrapper);
  auto lap_Phi = compute_laplacian(param.N, param.L, Phi);
  
  Eigen::VectorXd rho_eigen(rho.size());
  Eigen::VectorXd Phi_eigen(Phi.size());
  Eigen::VectorXd lap_Phi_eigen(lap_Phi.size());
  copy_vector(rho_eigen, rho);
  copy_vector(Phi_eigen, Phi);
  copy_vector(lap_Phi_eigen, lap_Phi);

  rho_eigen.array() -= rho_eigen.mean();

  // Phi_eigen /= param.N * param.N * param.N;
  // Phi_eigen.array() -= Phi_eigen.mean();

  std::cout << "rho_eigen.mean() = " << rho_eigen.mean() << '\n';
  std::cout << "lap_Phi_eigen.mean() = " << lap_Phi_eigen.mean() << '\n';
  
  std::cout << "rho_eigen.norm() = " << rho_eigen.norm() << '\n';
  std::cout << "Phi_eigen.norm() = " << Phi_eigen.norm() << '\n';
  std::cout << "lap_Phi_eigen.norm() = " << lap_Phi_eigen.norm() << '\n';
  std::cout << "diff.norm() = " << (rho_eigen - lap_Phi_eigen).norm() << '\n';

  std::cout << "rho = " << rho_eigen.head(16).transpose() << '\n';
  std::cout << "Phi = " << Phi_eigen.head(16).transpose() << '\n';
  std::cout << "lap_Phi = " << lap_Phi_eigen.head(16).transpose() << '\n';
  
  std::cout << "(rho_eigen - lap_Phi_eigen).norm() / rho_eigen.norm() = " << (rho_eigen - lap_Phi_eigen).norm() / rho_eigen.norm() << '\n';
  std::cout << "(rho_eigen - lap_Phi_eigen).L_inf_norm() / rho_eigen.L_inf_norm() = " << (rho_eigen - lap_Phi_eigen).lpNorm<Eigen::Infinity>() / rho_eigen.lpNorm<Eigen::Infinity>() << '\n';
  assert((rho_eigen - lap_Phi_eigen).norm() / rho_eigen.norm() < 1e-14);
}


void test_cuda_08(void)
{
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  Workspace workspace(param, plane_wave);

  auto rho = Equation::compute_energy_density(workspace, 0);
  auto Phi = compute_inverse_laplacian(param.N, param.L, rho, workspace.fft_wrapper);
  auto lap_Phi = compute_laplacian(param.N, param.L, Phi);

  auto fft = workspace.fft_wrapper.execute_d2z(rho);

  Eigen::VectorXd state_eigen(workspace.state.size());
  Eigen::VectorXd rho_eigen(rho.size());
  Eigen::VectorXd Phi_eigen(Phi.size());
  Eigen::VectorXd lap_Phi_eigen(lap_Phi.size());
  Eigen::VectorXd fft_eigen(fft.size());
  copy_vector(state_eigen, workspace.state);
  copy_vector(rho_eigen, rho);
  copy_vector(Phi_eigen, Phi);
  copy_vector(lap_Phi_eigen, lap_Phi);
  copy_vector(fft_eigen, fft);

  rho_eigen.array() -= rho_eigen.mean();
  lap_Phi_eigen.array() -= lap_Phi_eigen.mean();

  // Phi_eigen /= param.N * param.N * param.N;
  // Phi_eigen.array() -= Phi_eigen.mean();
  std::cout << "rho_eigen.mean() = " << rho_eigen.mean() << '\n';
  std::cout << "lap_Phi_eigen.mean() = " << lap_Phi_eigen.mean() << '\n';
  
  std::cout << "rho_eigen.norm() = " << rho_eigen.norm() << '\n';
  std::cout << "Phi_eigen.norm() = " << Phi_eigen.norm() << '\n';
  std::cout << "lap_Phi_eigen.norm() = " << lap_Phi_eigen.norm() << '\n';
  std::cout << "diff.norm() = " << (rho_eigen - lap_Phi_eigen).norm() << '\n';
  
  std::cout << "state = " << state_eigen.head(16).transpose() << '\n';
  std::cout << "rho = " << rho_eigen.head(16).transpose() << '\n';
  std::cout << "Phi = " << Phi_eigen.head(16).transpose() << '\n';
  std::cout << "lap_Phi = " << lap_Phi_eigen.head(16).transpose() << '\n';
  std::cout << "fft = " << fft_eigen.head(16).transpose() << '\n';

  std::cout << "(rho_eigen - lap_Phi_eigen).norm() / rho_eigen.norm() = " << (rho_eigen - lap_Phi_eigen).norm() / rho_eigen.norm() << '\n';
  assert((rho_eigen - lap_Phi_eigen).norm() / rho_eigen.norm() < 1e-14);
}

void test_cuda_09(void)
{
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  Workspace workspace(param, plane_wave);

  Eigen::VectorXd f(param.N * param.N * param.N);
  thrust::copy(workspace.state.begin(), workspace.state.begin() + f.size(), f.begin());
  auto Phi = compute_inverse_laplacian(param.N, param.L, workspace.state, workspace.fft_wrapper);
  auto lap_Phi = compute_laplacian(param.N, param.L, Phi);

  auto fft = workspace.fft_wrapper.execute_d2z(workspace.state);
  auto Phi_fft = workspace.fft_wrapper.execute_d2z(Phi);

  Eigen::VectorXd Phi_eigen(Phi.size());
  Eigen::VectorXd lap_Phi_eigen(lap_Phi.size());
  Eigen::VectorXd fft_eigen(fft.size());
  Eigen::VectorXd Phi_fft_eigen(Phi_fft.size());
  copy_vector(Phi_eigen, Phi);
  copy_vector(lap_Phi_eigen, lap_Phi);
  copy_vector(fft_eigen, fft);
  copy_vector(Phi_fft_eigen, Phi_fft);

  
  std::cout << "f.norm() = " << f.norm() << '\n';
  std::cout << "Phi_eigen.norm() = " << Phi_eigen.norm() << '\n';
  std::cout << "lap_Phi_eigen.norm() = " << lap_Phi_eigen.norm() << '\n';
  std::cout << "diff.norm() = " << (f - lap_Phi_eigen).norm() << '\n';
  
  std::cout << "f = " << f.head(16).transpose() << '\n';
  std::cout << "Phi = " << Phi_eigen.head(16).transpose() << '\n';
  std::cout << "lap_Phi = " << lap_Phi_eigen.head(16).transpose() << '\n';
  std::cout << "fft = " << fft_eigen.head(16).transpose() << '\n';
  std::cout << "Phi_fft = " << Phi_fft_eigen.head(16).transpose() << '\n';

  std::cout << "(f - lap_Phi_eigen).norm() / f.norm() = " << (f - lap_Phi_eigen).norm() / f.norm() << '\n';
  
  assert((f - lap_Phi_eigen).norm() / f.norm() < 1e-14);
}


void test_cuda_10(void)
{
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;

  MyParam param;
  Workspace workspace(param, plane_wave);

  Eigen::VectorXd f(param.N * param.N * param.N);
  thrust::copy(workspace.state.begin(), workspace.state.begin() + f.size(), f.begin());
  //auto Phi = compute_inverse_laplacian(param.N, param.L, workspace.state, workspace.fft_wrapper);

  auto fft = workspace.fft_wrapper.execute_d2z(workspace.state);
  Eigen::VectorXd fft_1_eigen(fft.size());
  Eigen::VectorXd fft_2_eigen(fft.size());

  copy_vector(fft_1_eigen, fft);
  compute_inverse_laplacian_test(param.N, param.L, fft);
  copy_vector(fft_2_eigen, fft);

  std::cout << "fft_1 = " << fft_1_eigen.head(16).transpose() << '\n';
  std::cout << "fft_2 = " << fft_2_eigen.head(16).transpose() << '\n';
}

void test_cuda_11(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  
  MyParam2 param;
  print_param(param);
  
  typedef CudaSqrtPotentialEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, unperturbed_grf);
  Equation eqn(workspace);
  const_interval_observer observer(std::string("../../Data/Sqrt_Potential_Test/"), param, eqn);
  auto stepper = runge_kutta4<State, double, State, double>();

  stepper.do_step(eqn, workspace.state, 0.0, 0.01);

  Eigen::VectorXd f(workspace.state.size());
  copy_vector(f, workspace.state);
  // //thrust::copy(workspace.state.begin(), workspace.state.begin() + f.size(), f.begin());

  std::cout << "f.head = " << f.head(16).transpose() << '\n';
  std::cout << "f.tail = " << f.tail(16).transpose() << '\n';
}

struct MyParam3 {
  // lattice params
  long long int N = 384;
  double L = 384 * 50.0;
  // ULDM params
  double m = 1.0;
  double k_ast = 0.1;
  double k_Psi = 0.03;
  double varphi_std_dev = 1.0;
  double Psi_std_dev = 0.15;
  // FRW metric params
  double a1 = 1.0;
  double H1 = 0.005;
  double t1 = 1.0 / (2 * 0.005);
  // // Solution record params
  // double t_start = param.t1;
  // double t_end = param.t_start + (pow(3.0 / param.a1, 2) - 1.0) / (2 * param.H1);
  // double t_interval = 10;
  // Solution record params
  double t_start = 1.0 / (2 * 0.005);
  double t_end = 1.0 / (2 * 0.005) + 10.0;
  double t_interval = 10.0;
};

void test_cuda_12(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  
  MyParam3 param;
  print_param(param);
  
  typedef CudaComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, perturbed_grf_and_comoving_curvature_fft);
  Equation eqn(workspace);
  const_interval_observer observer(std::string("../../Data/Comoving_Curvature_Test/"), param, eqn);
  auto stepper = runge_kutta4<State, double, State, double>();

  Eigen::VectorXd R_fft_eigen(workspace.R_fft.size());
  copy_vector(R_fft_eigen, workspace.R_fft);
  std::cout << "R_fft_eigen.head = " << R_fft_eigen.head(16).transpose() << '\n';


  // {
  //   const double t = param.t1;
  //   const auto N = workspace.N;
  //   const auto L = workspace.L;
  //   const auto m = workspace.m;
  //   const auto a_t = workspace.cosmology.a(t);
  //   const auto H_t = workspace.cosmology.H(t);
  //   const auto eta_t = workspace.cosmology.eta(t);
  //   const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  //   // std::cout << "===============\n";
  //   // std::cout << "t = " << t << '\n';
  //   // std::cout << "N = " << N << '\n';
  //   // std::cout << "L = " << L << '\n';
  //   // std::cout << "m = " << m << '\n';
  //   // std::cout << "a_t = " << a_t << '\n';
  //   // std::cout << "H_t = " << H_t << '\n';
  //   // std::cout << "eta_t = " << eta_t << '\n';
  //   // std::cout << "inv_ah_sqr = " << inv_ah_sqr << '\n';
  //   // std::cout << "===============\n";

  //   thrust::device_vector<double> Psi(N * N * N);
  //   kernel_test(workspace.R_fft, Psi, N, L, m, a_t, H_t, eta_t, inv_ah_sqr, t, workspace.fft_wrapper);
  //   Eigen::VectorXd Psi_eigen(Psi.size());
  //   copy_vector(Psi_eigen, Psi);
  //   std::cout << "Psi.head = " << Psi_eigen.head(16).transpose() << '\n';
  // }
  
  std::cout << "point 1\n";
  stepper.do_step(eqn, workspace.state, param.t1, 0.01);
  
  Eigen::VectorXd f(workspace.state.size());
  copy_vector(f, workspace.state);
  // //thrust::copy(workspace.state.begin(), workspace.state.begin() + f.size(), f.begin());

  std::cout << "f.head = " << f.head(16).transpose() << '\n';
  std::cout << "f.tail = " << f.tail(16).transpose() << '\n';
}


void test_cuda_13(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  
  MyParam2 param;
  print_param(param);
  
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, unperturbed_grf);
  Equation eqn(workspace);
  //const_interval_observer observer(std::string("../../Data/Klein_Gordon_Test/"), param, eqn);
  auto stepper = runge_kutta4<State, double, State, double>();

  const double delta_t = 0.00001;
  
  Eigen::VectorXd dot_rho_eigen(workspace.state.size() / 2);
  Eigen::VectorXd rho_0_eigen(workspace.state.size() / 2);
  Eigen::VectorXd rho_1_eigen(workspace.state.size() / 2);

  {
    auto dot_rho = Equation::compute_dot_energy_density(eqn.workspace, 0.0);
    copy_vector(dot_rho_eigen, dot_rho);
  }
  
  {
    auto rho_0 = Equation::compute_energy_density(eqn.workspace, 0.0);
    copy_vector(rho_0_eigen, rho_0);
  }
  
  stepper.do_step(eqn, workspace.state, 0.0, delta_t);

  {
    auto rho_1 = Equation::compute_energy_density(eqn.workspace, 0.0);
    copy_vector(rho_1_eigen, rho_1);
  }

  auto dot_rho_eigen_2 = (rho_1_eigen - rho_0_eigen) / delta_t;

  double diff = (dot_rho_eigen - dot_rho_eigen_2).norm() / dot_rho_eigen_2.norm();
  
  std::cout << "analytic = " << dot_rho_eigen.head(16).transpose() << '\n';
  std::cout << "numeric = " << dot_rho_eigen_2.head(16).transpose() << '\n';
  std::cout << "diff = " << diff << '\n';
}


void test_cuda_14(void)
{
  using namespace boost::numeric::odeint;
  using namespace Eigen;
  using namespace std::numbers;
  RandomNormal::set_generator_seed(0);
  
  MyParam2 param;
  param.N = 384;
  param.L = 384 * 0.8;
  param.m = 1.0;
  param.k_ast = 1.0;
  param.k_Psi = 1.0;
  param.varphi_std_dev = 1.0;
  param.Psi_std_dev = 0.2;
  param.a1 = 1.0;
  param.H1 = 0.05;
  param.t1 = 1 / (2 * param.H1);
  param.t_start = param.t1;
  print_param(param);

  IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  
  typedef CudaComovingCurvatureEquationInFRW CudaEquation;
  typedef typename CudaEquation::Workspace CudaWorkspace;
  typedef typename CudaEquation::State CudaState;
  CudaWorkspace cuda_workspace(param, perturbed_grf_and_comoving_curvature_fft);
  CudaEquation cuda_eqn(cuda_workspace);
  
  typedef ComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, homogeneous_field);
  Equation eqn(workspace);

  workspace.R_fft.resize(cuda_workspace.R_fft.size());
  copy_vector(workspace.state, cuda_workspace.state);
  copy_vector(workspace.R_fft, cuda_workspace.R_fft);
  
  Eigen::VectorXd rho(workspace.state.size() / 2);
  Eigen::VectorXd cuda_rho(workspace.state.size() / 2);

  rho = Equation::compute_energy_density(workspace, param.t1);
  
  {
    auto cuda_rho_thrust = CudaEquation::compute_energy_density(cuda_workspace, param.t1);
    copy_vector(cuda_rho, cuda_rho_thrust);
  }

  double diff = (rho - cuda_rho).norm() / cuda_rho.norm();

  std::cout << "rho = " << rho.tail(16).transpose().format(HeavyFmt) << '\n';
  std::cout << "cuda_rho = " << cuda_rho.tail(16).transpose().format(HeavyFmt) << '\n';
  std::cout << "diff = " << diff << '\n';


  const double t = param.t1;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  
  thrust::device_vector<double> cuda_Psi_thrust(cuda_workspace.R_fft.size());
  thrust::device_vector<double> cuda_dPsidt_thrust(cuda_workspace.R_fft.size());
  kernel_test(cuda_workspace.R_fft, cuda_Psi_thrust, cuda_dPsidt_thrust, param.N, param.L, param.m,
  	      a_t, H_t, eta_t, inv_ah_sqr, t, cuda_workspace.fft_wrapper);
  
  //IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  Eigen::VectorXd cuda_Psi(cuda_Psi_thrust.size());
  copy_vector(cuda_Psi, cuda_Psi_thrust);
  std::cout << "cuda_Psi = " << cuda_Psi.head(16).transpose().format(HeavyFmt) << '\n';
  
  
  // {
  //  IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  //   Eigen::VectorXd cuda_state(workspace.state.size());
  //   copy_vector(cuda_state, workspace.state);
  //   std::cout << "state = " << workspace.state.tail(16).transpose() << '\n';
  //   std::cout << "cuda_state = " << cuda_state.tail(16).transpose() << '\n';
  //   double state_diff = (workspace.state - cuda_state).norm() / cuda_state.norm();
  //   std::cout << "state_diff = " << state_diff << '\n';
  // }
}


void test_cuda_17(void)
{
  using namespace boost::numeric::odeint;
  using namespace Eigen;
  using namespace std::numbers;
  RandomNormal::set_generator_seed(0);
  
  MyParam2 param;
  param.N = 384;
  param.L = 384 * 0.8;
  param.m = 1.0;
  param.k_ast = 1.0;
  param.k_Psi = 1.0;
  param.varphi_std_dev = 1.0;
  param.Psi_std_dev = 0.1;
  param.a1 = 1.0;
  param.H1 = 0.05;
  param.t1 = 1 / (2 * param.H1);
  param.t_start = param.t1;
  print_param(param);


  
  IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  
  typedef CudaComovingCurvatureEquationInFRW CudaEquation;
  typedef typename CudaEquation::Workspace CudaWorkspace;
  typedef typename CudaEquation::State CudaState;
  CudaWorkspace cuda_workspace(param, perturbed_grf_and_comoving_curvature_fft);
  CudaEquation cuda_eqn(cuda_workspace);
  
  typedef ComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, homogeneous_field);
  Equation eqn(workspace);


  const double t = param.t1;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  
  workspace.R_fft.resize(cuda_workspace.R_fft.size());
  copy_vector(workspace.R_fft, cuda_workspace.R_fft);

  copy_vector(workspace.state, cuda_workspace.state);
  // workspace.state.head(N*N*N) = Eigen::VectorXd::Constant(N*N*N, 1.0);
  // workspace.state.tail(N*N*N) = Eigen::VectorXd::Constant(N*N*N, 0);
  //thrust::copy(workspace.state.begin(), workspace.state.end(), cuda_workspace.state.begin());

  
  Eigen::VectorXd dxdt(workspace.state.size());
  Eigen::VectorXd cuda_dxdt(workspace.state.size());

  eqn(workspace.state, dxdt, param.t1);
  
  {
    thrust::device_vector<double> cuda_dxdt_thrust(workspace.state.size());
    cuda_eqn(cuda_workspace.state, cuda_dxdt_thrust, param.t1);
    copy_vector(cuda_dxdt, cuda_dxdt_thrust);
  }

  int length = N;
  // std::cout << "dxdt = " << dxdt(seqN(N*N*N+N,16)).transpose().format(HeavyFmt) << '\n';
  // std::cout << "cuda_dxdt = " << cuda_dxdt(seqN(N*N*N+N,16)).transpose().format(HeavyFmt) << '\n';
  std::cout << "dxdt = " << dxdt.tail(N).transpose() << '\n';
  std::cout << "cuda_dxdt = " << cuda_dxdt.tail(N).transpose() << '\n';
  std::cout << "diff = " << (dxdt - cuda_dxdt).norm() / cuda_dxdt.norm() << '\n';
  std::cout << "diff1 = " << (dxdt - cuda_dxdt).tail(length).norm() << '\n';
  std::cout << "diff2 = " << dxdt.tail(length).norm() << '\n';
  std::cout << "diff3 = " << cuda_dxdt.tail(length).norm() << '\n';
  std::cout << '\n';


  // {
  //   Eigen::VectorXd cuda_Psi_padded(cuda_Psi_thrust.size());
  //   copy_vector(cuda_Psi_padded, cuda_Psi_thrust);
  //   Eigen::VectorXd cuda_Psi(N*N*N); //Eigen::VectorXd::NullaryExpr(N*N,[&](){return dis(gen);});;
  //   for(int a = 0; a < N; ++a){
  //     for(int b = 0; b < N; ++b){
  // 	for(int c = 0; c < N; ++c){
  // 	  cuda_Psi(IDX_OF(N, a, b, c)) = cuda_Psi_padded(c + (N/2+1)*2*b + N*(N/2+1)*2*a);
  // 	}
  //     }
  //   }
  //   std::cout << "cuda_Psi = " << cuda_Psi.tail(16).transpose() << '\n';
  //   std::cout << "Psi = " << workspace.Psi.tail(16).transpose() << '\n';
  //   std::cout << "diff = " << (workspace.Psi - cuda_Psi).norm() / cuda_Psi.norm() << '\n';
  //   std::cout << '\n';
  // }

  // thrust::device_vector<double> cuda_Psi_thrust(cuda_workspace.R_fft.size());
  // thrust::device_vector<double> cuda_dPsidt_thrust(cuda_workspace.R_fft.size());
  // thrust::device_vector<double> cuda_Psi_thrust;
  // thrust::device_vector<double> cuda_dPsidt_thrust;
  // kernel_test(cuda_workspace.R_fft, cuda_Psi_thrust, cuda_dPsidt_thrust, param.N, param.L, param.m,
  // 	      a_t, H_t, eta_t, inv_ah_sqr, t, cuda_workspace.fft_wrapper);
  
  //IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  // {
  //   Eigen::VectorXd cuda_Psi(cuda_Psi_thrust.size());
  //   copy_vector(cuda_Psi, cuda_Psi_thrust);
  //   std::cout << "cuda_Psi = " << cuda_Psi.tail(N).transpose().format(HeavyFmt) << '\n';
  //   std::cout << "Psi = " << workspace.Psi.tail(N).transpose().format(HeavyFmt) << '\n';
  //   std::cout << "diff = " << (workspace.Psi - cuda_Psi).norm() / cuda_Psi.norm() << '\n';
  //   std::cout << '\n';
  // }

  // {
  //   Eigen::VectorXd cuda_dPsidt(cuda_dPsidt_thrust.size());
  //   copy_vector(cuda_dPsidt, cuda_dPsidt_thrust);
  //   std::cout << "cuda_dPsidt = " << cuda_dPsidt.head(16).transpose().format(HeavyFmt) << '\n';
  //   std::cout << "dPsidt = " << workspace.dPsidt.head(16).transpose().format(HeavyFmt) << '\n';
  //   std::cout << "diff = " << (workspace.dPsidt - cuda_dPsidt).norm() / cuda_dPsidt.norm() << '\n';
  //   std::cout << '\n';
  // }
  
}


// Test padded FFTW format
void test_cuda_18(void)
{
  using namespace boost::numeric::odeint;
  using namespace Eigen;
  using namespace std::numbers;
  RandomNormal::set_generator_seed(0);
  
  MyParam2 param;
  param.N = 384;
  param.L = 384 * 0.8;
  param.m = 1.0;
  param.k_ast = 1.0;
  param.k_Psi = 1.0;
  param.varphi_std_dev = 1.0;
  param.Psi_std_dev = 0.1;
  param.a1 = 1.0;
  param.H1 = 0.05;
  param.t1 = 1 / (2 * param.H1);
  param.t_start = param.t1;
  print_param(param);


  
  IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  
  typedef CudaComovingCurvatureEquationInFRW CudaEquation;
  typedef typename CudaEquation::Workspace CudaWorkspace;
  typedef typename CudaEquation::State CudaState;
  CudaWorkspace cuda_workspace(param, perturbed_grf_and_comoving_curvature_fft);
  CudaEquation cuda_eqn(cuda_workspace);
  
  typedef ComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, homogeneous_field);
  Equation eqn(workspace);


  const double t = param.t1;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  
  workspace.R_fft.resize(cuda_workspace.R_fft.size());
  copy_vector(workspace.R_fft, cuda_workspace.R_fft);

  copy_vector(workspace.state, cuda_workspace.state);
  // workspace.state.head(N*N*N) = Eigen::VectorXd::Constant(N*N*N, 1.0);
  // workspace.state.tail(N*N*N) = Eigen::VectorXd::Constant(N*N*N, 0);
  //thrust::copy(workspace.state.begin(), workspace.state.end(), cuda_workspace.state.begin());

  Eigen::VectorXd dxdt(workspace.state.size());
  eqn(workspace.state, dxdt, param.t1);
  
  thrust::device_vector<double> cuda_Psi_thrust(cuda_workspace.R_fft.size());
  thrust::device_vector<double> cuda_dPsidt_thrust(cuda_workspace.R_fft.size());
  // thrust::device_vector<double> cuda_Psi_thrust;
  // thrust::device_vector<double> cuda_dPsidt_thrust;
  kernel_test(cuda_workspace.R_fft, cuda_Psi_thrust, cuda_dPsidt_thrust, param.N, param.L, param.m,
  	      a_t, H_t, eta_t, inv_ah_sqr, t, cuda_workspace.fft_wrapper);
  
  //IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  {
    Eigen::VectorXd cuda_Psi_padded(cuda_Psi_thrust.size());
    copy_vector(cuda_Psi_padded, cuda_Psi_thrust);
    Eigen::VectorXd cuda_Psi(N*N*N); //Eigen::VectorXd::NullaryExpr(N*N,[&](){return dis(gen);});;
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  cuda_Psi(IDX_OF(N, a, b, c)) = cuda_Psi_padded(c + (N/2+1)*2*b + N*(N/2+1)*2*a);
	}
      }
    }
    std::cout << "cuda_Psi = " << cuda_Psi.tail(16).transpose() << '\n';
    std::cout << "Psi = " << workspace.Psi.tail(16).transpose() << '\n';
    std::cout << "diff = " << (workspace.Psi - cuda_Psi).norm() / cuda_Psi.norm() << '\n';
    std::cout << '\n';
  }

}


/*
void test_cuda_14(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  
  const std::string dir = "../../Data/Dot_Delta_Test_1/";
  prepare_directory_for_output(dir);
  
  MyParam2 param;
  param.L = param.N * 4.0;
  param.Psi_std_dev = 0;
  print_param(param);
  save_param_for_Mathematica(param, dir);
  
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  //Workspace workspace(param, unperturbed_grf);
  Workspace workspace(param, perturbed_grf_in_radiation_domination);
  Equation eqn(workspace);
  //const_interval_observer observer(dir, param, eqn);
  //auto stepper = runge_kutta4<State, double, State, double>();

  const double delta_t = 0.00001;
  
  // Eigen::VectorXd dot_rho_eigen(workspace.state.size() / 2);
  // Eigen::VectorXd rho_0_eigen(workspace.state.size() / 2);
  // Eigen::VectorXd rho_1_eigen(workspace.state.size() / 2);


  {
    auto dot_rho = Equation::compute_dot_energy_density(eqn.workspace, 0.0);
    auto dot_rho_spectrum = compute_power_spectrum(param.N, dot_rho, workspace.fft_wrapper);
    Eigen::VectorXd dot_rho_spectrum_out(dot_rho_spectrum.size());
    copy_vector(dot_rho_spectrum_out, dot_rho_spectrum);
    write_VectorXd_to_filename_template(dot_rho_spectrum_out, dir + "dot_rho_spectrum_%d.dat", 0);
  }
  {
    auto rho = Equation::compute_energy_density(eqn.workspace, 0.0);
    auto rho_spectrum = compute_power_spectrum(param.N, rho, workspace.fft_wrapper);
    Eigen::VectorXd rho_spectrum_out(rho_spectrum.size());
    copy_vector(rho_spectrum_out, rho_spectrum);
    write_VectorXd_to_filename_template(rho_spectrum_out, dir + "rho_spectrum_%d.dat", 0);
  }
}
*/


// Try adding a phase to get velocity
void test_cuda_15(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  
  const std::string dir = "../../Data/Dot_Delta_Test_7/";
  prepare_directory_for_output(dir);
  
  MyParam2 param;
  param.L = param.N * 0.4;
  param.k_ast = 1.0;
  
  param.t_start = 0;
  param.t_end = 30;
  param.t_interval = 1;
  print_param(param);
  save_param_for_Mathematica(param, dir);
  
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, unperturbed_grf);
  Equation eqn(workspace);
  
  Eigen::VectorXd new_state(workspace.state.size());
  copy_vector(new_state, workspace.state);

  {
    const long long int N = param.N;
    Eigen::VectorXd phase(N*N*N);
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  phase(IDX_OF(N, a, b, c)) = 0.5 * std::numbers::pi * cos(2 * std::numbers::pi * b / N);
	}
      }
    }

    // Eigen::VectorXd varphi = new_state.head(N*N*N);
    // Eigen::VectorXd dt_varphi = new_state.tail(N*N*N);
  
    // boost_klein_gordon_field(varphi, dt_varphi, phase, N, param.L, param.m);
    // new_state.head(N*N*N) = varphi;
    // new_state.tail(N*N*N) = dt_varphi;
  
    add_phase_to_state(new_state, phase);
    thrust::copy(new_state.begin(), new_state.end(), workspace.state.begin());
  }
  
  //const_interval_observer observer(dir, param, eqn);
  const_interval_observer<Equation, true, true, true> observer(dir, param, eqn);
  auto stepper = runge_kutta4<State, double, State, double>();
  run_and_measure_time("Solving equation",
  		       [&](){ integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, 0.05 / param.m, observer); } );
  
  write_vector_to_file(workspace.t_list, dir + "t_list.dat");
}


void test_cuda_16(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  
  const std::string dir = "../../Data/Dot_Delta_Test_4/";
  prepare_directory_for_output(dir);
  
  MyParam2 param;
  param.L = param.N * 4.0;
  param.t_start = 0;
  param.t_end = 1000;
  param.t_interval = 10;
  print_param(param);
  save_param_for_Mathematica(param, dir);
  
  typedef CudaKleinGordonEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, unperturbed_grf);
  Equation eqn(workspace);
  
  Eigen::VectorXd new_state(workspace.state.size());
  copy_vector(new_state, workspace.state);

  {
    
    const long long int N = param.N;
    Eigen::VectorXd minus_dot_delta(N*N*N);
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  if(c > N / 3 && c < 2 * N / 3 && b > N / 3 && b < 2 * N / 3) {
	    minus_dot_delta(IDX_OF(N, a, b, c)) = -0.0005;
	  } else {
	    minus_dot_delta(IDX_OF(N, a, b, c)) = 0;
	  }
	}
      }
    }
  
    fftWrapperDispatcher<Eigen::VectorXd>::Generic eigen_fft_wrapper(N);
    Eigen::VectorXd phase = compute_inverse_laplacian(N, param.L, minus_dot_delta, eigen_fft_wrapper);

    // std::cout << "phase.max() = " << phase.maxCoeff() << '\n';
    // std::cout << "phase.min() = " << phase.minCoeff() << '\n';

    Eigen::VectorXd varphi = new_state.head(N*N*N);
    Eigen::VectorXd dt_varphi = new_state.tail(N*N*N);
  
    boost_klein_gordon_field(varphi, dt_varphi, phase, N, param.L, param.m);
    new_state.head(N*N*N) = varphi;
    new_state.tail(N*N*N) = dt_varphi;
  
    //add_phase_to_state(new_state, phase);
    thrust::copy(new_state.begin(), new_state.end(), workspace.state.begin());

  }

  //const_interval_observer observer(dir, param, eqn);
  const_interval_observer<Equation, true, true, true> observer(dir, param, eqn);
  auto stepper = runge_kutta4<State, double, State, double>();
  run_and_measure_time("Solving equation",
  		       [&](){ integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, 0.1 / param.m, observer); } );
  
  write_vector_to_file(workspace.t_list, dir + "t_list.dat");
}

void test_cuda_19(void)
{
  test_texture();
}

#endif

// Learn to use FFTW's in-place transform
void test_00(void)
{
  using namespace Eigen;
  int N = 256;
  fftwWrapper fft_wrapper(N);
  Eigen::VectorXd f = Eigen::VectorXd::Random(N * N * N);
  Eigen::VectorXd f_fft = fft_wrapper.execute_d2z(f) / (N*N*N);
  
  Eigen::VectorXd in_place = f_fft;
  fft_wrapper.execute_inplace_z2d(in_place);
  
  Eigen::VectorXd out_place = fft_wrapper.execute_z2d(f_fft);
  

  // Compare
  {
    int offset = N * N * N;
    IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    // std::cout << "out_place = " << out_place(seqN(N,10)).transpose().format(HeavyFmt) << '\n';
    // std::cout << "in_place = " << in_place(seqN(N,10)).transpose().format(HeavyFmt) << '\n';
    std::cout << "out_place = " << out_place.head(10).transpose().format(HeavyFmt) << '\n';
    std::cout << "in_place = " << in_place.head(10).transpose().format(HeavyFmt) << '\n';
    double diff = (out_place.head(offset) - in_place.head(offset)).norm() / out_place.head(offset).norm();
    std::cout << "diff = " << diff << '\n';
  }
}

// Test whether ODEINT stepper can step backward
void test_01(void)
{
  using namespace boost::numeric::odeint;
  RandomNormal::set_generator_seed(0);
  typedef std::array<double, 2> State;
  auto stepper = runge_kutta4<State>();
  auto eqn = [](const State &x, State &dxdt, const double) {
	       dxdt[0] = x[1];
	       dxdt[1] = -x[0];
	     };

  State state({1, 0});
  double delta_t = 0.1;
  stepper.do_step(eqn, state, 0.0, -delta_t);

  std::cout << "soln = " << state[0] << ", " << state[1] << '\n';
  std::cout << "soln = " << std::cos(delta_t) << ", " << std::sin(delta_t) << '\n';
}

// Measure FFTW performance
void test_02(void)
{
  using namespace Eigen;
  int N = 384;
  fftwWrapper fft_wrapper(N);
  Eigen::VectorXd f = Eigen::VectorXd::Random(N * N * N);
  Eigen::VectorXd f_fft = fft_wrapper.execute_d2z(f) / (N*N*N);
  
  // Eigen::VectorXd in_place = f_fft;
  // fft_wrapper.execute_inplace_z2d(in_place);
  
  // Eigen::VectorXd out_place = fft_wrapper.execute_z2d(f_fft);

  run_and_measure_time("out of place FFT", [&](){ fft_wrapper.execute_z2d(f_fft); });
  
  f_fft = fft_wrapper.execute_d2z(f) / (N*N*N);
  
  run_and_measure_time("in place FFT", [&](){ fft_wrapper.execute_inplace_z2d(f_fft); });
}


void test_03(void)
{
  using namespace boost::numeric::odeint;
  using namespace Eigen;
  using namespace std::numbers;
  RandomNormal::set_generator_seed(0);
  
  MyParam2 param;
  param.N = 384;
  param.L = 384 * 0.8;
  param.m = 1.0;
  param.k_ast = 1.0;
  param.k_Psi = 1.0;
  param.varphi_std_dev = 1.0;
  param.Psi_std_dev = 0.02;
  param.a1 = 1.0;
  param.H1 = 0.05;
  param.t1 = 1 / (2 * param.H1);
  param.t_start = param.t1;
  print_param(param);

  
  IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  
  typedef ComovingCurvatureEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  Workspace workspace(param, perturbed_grf_and_comoving_curvature_fft);
  Equation eqn(workspace);


  const double t = param.t1;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  const int scale_diff = 3;
  const int M = 384 / scale_diff;

  const double t_eval = 1 * param.t1;
  Eigen::VectorXd dxdt(workspace.state.size());
  eqn(workspace.state, dxdt, t_eval);

  const double k_UV_M = (2 * pi / L) * (M / 2);
  std::cout << "k_UV_M eta_i = " << k_UV_M * eta_t << '\n';
  std::cout << "k_UV_M eta_eval = " << k_UV_M * workspace.cosmology.eta(t_eval) << '\n';
  
  // Eigen::VectorXd reduced_R_fft(2 * M * M * (M / 2 + 1));
  // for(int a = 0; a < M; ++a){
  //   for(int b = 0; b < M; ++b){
  //     for(int c = 0; c <= M/2; ++c){
  // 	int a_N = (a<M/2) ? a : (N-(M-a));
  // 	int b_N = (b<M/2) ? b : (N-(M-b));
  // 	int c_N = c;
	
  // 	int idx_M = M*(M/2+1)*a + (M/2+1)*b + c;
  // 	int idx_N = N*(N/2+1)*a_N + (N/2+1)*b_N + c_N;
  // 	reduced_R_fft(2 * idx_M) = workspace.R_fft(2 * idx_N) / pow(scale_diff, 3);
  // 	reduced_R_fft(2 * idx_M + 1) = workspace.R_fft(2 * idx_N + 1) / pow(scale_diff, 3);
  //     }
  //   }
  // }

  Eigen::VectorXd reduced_R_fft = compute_cutoff_fouriers(N, M, workspace.R_fft);
  
  {
    // Save spectrum for initial potential Psi
    double eta_i = workspace.cosmology.eta(t_eval);
    auto kernel = [eta_i](double k){
		    return k == 0.0 ? 0.0 : (6 * sqrt(3) * (-((k * eta_i * cos((k * eta_i) / sqrt(3))) / sqrt(3)) + sin((k * eta_i) / sqrt(3)))) / (pow(k, 3) * pow(eta_i, 3));
		  };
    
    auto fft_wrapper = fftwWrapper(M);
    Eigen::VectorXd reduced_R = fft_wrapper.execute_z2d(reduced_R_fft) / pow(M, 3);
    Eigen::VectorXd reduced_Psi = compute_field_with_scaled_fourier_modes(M, param.L, reduced_R, kernel, fft_wrapper);

    Eigen::VectorXd interpolated_Psi(N*N*N);
    Eigen::VectorXd copied_Psi_1(N*N*N);
    Eigen::VectorXd copied_Psi_2(N*N*N);
    
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  // Trilinear interpolation
	  double val_000 = reduced_Psi(IDX_OF(M, a/scale_diff, b/scale_diff, c/scale_diff));
	  double val_001 = reduced_Psi(IDX_OF(M, a/scale_diff, b/scale_diff, (c/scale_diff+1)%M));
	  double val_010 = reduced_Psi(IDX_OF(M, a/scale_diff, (b/scale_diff+1)%M, c/scale_diff));
	  double val_011 = reduced_Psi(IDX_OF(M, a/scale_diff, (b/scale_diff+1)%M, (c/scale_diff+1)%M));
	  double val_100 = reduced_Psi(IDX_OF(M, (a/scale_diff+1)%M, b/scale_diff, c/scale_diff));
	  double val_101 = reduced_Psi(IDX_OF(M, (a/scale_diff+1)%M, b/scale_diff, (c/scale_diff+1)%M));
	  double val_110 = reduced_Psi(IDX_OF(M, (a/scale_diff+1)%M, (b/scale_diff+1)%M, c/scale_diff));
	  double val_111 = reduced_Psi(IDX_OF(M, (a/scale_diff+1)%M, (b/scale_diff+1)%M, (c/scale_diff+1)%M));

	  double coeff_a = (a - scale_diff*(a/scale_diff)) / double(scale_diff);
	  double coeff_b = (b - scale_diff*(b/scale_diff)) / double(scale_diff);
	  double coeff_c = (c - scale_diff*(c/scale_diff)) / double(scale_diff);
	  
	  double val_00 = (1-coeff_c) * val_000 + coeff_c * val_001;
	  double val_01 = (1-coeff_c) * val_010 + coeff_c * val_011;
	  double val_10 = (1-coeff_c) * val_100 + coeff_c * val_101;
	  double val_11 = (1-coeff_c) * val_110 + coeff_c * val_111;

	  double val_0 = (1-coeff_b) * val_00 + coeff_b * val_01;
	  double val_1 = (1-coeff_b) * val_10 + coeff_b * val_11;
	  
	  double val = (1-coeff_a) * val_0 + coeff_a * val_1;
	  
	  interpolated_Psi(IDX_OF(N, a, b, c)) = val;

	  copied_Psi_1(IDX_OF(N, a, b, c)) = reduced_Psi(IDX_OF(M, a/scale_diff, b/scale_diff, c/scale_diff));
	  copied_Psi_2(IDX_OF(N, (a+N-scale_diff/2)%N, (b+N-scale_diff/2)%N, (c+N-scale_diff/2)%N)) = reduced_Psi(IDX_OF(M, a/scale_diff, b/scale_diff, c/scale_diff));
	}
      }
    }
    
    std::cout << "workspace.Psi = " << workspace.Psi.head(16).transpose() << '\n';    
    std::cout << "interpolated_Psi = " << interpolated_Psi.head(16).transpose() << '\n';
    std::cout << "copied_Psi_1 = " << copied_Psi_1.head(16).transpose() << '\n';
    std::cout << "copied_Psi_2 = " << copied_Psi_2.head(16).transpose() << '\n';
    std::cout << "diff_interpolated = " << (interpolated_Psi - workspace.Psi).norm() / workspace.Psi.norm() << '\n';
    std::cout << "diff_copied_1 = " << (copied_Psi_1 - workspace.Psi).norm() / workspace.Psi.norm() << '\n';
    std::cout << "diff_copied_2 = " << (copied_Psi_2 - workspace.Psi).norm() / workspace.Psi.norm() << '\n';
    std::cout << "workspace.Psi.norm() = " << workspace.Psi.norm() << '\n';
    std::cout << "interpolated_Psi.norm() = " << interpolated_Psi.norm() << '\n';

    std::cout << "workspace.Psi min = " << workspace.Psi.minCoeff() << '\n';
    std::cout << "workspace.Psi max = " << workspace.Psi.maxCoeff() << '\n';
    
    std::cout << '\n';
  }
}



void test_04(void)
{
  using namespace boost::numeric::odeint;
  using namespace Eigen;
  using namespace std::numbers;
  RandomNormal::set_generator_seed(0);

  const double L = 384 * 0.8;
  const int N = 384;
  const int scale_diff = 3;
  const int M = N / scale_diff;
  const double k_ast = 0.5;
  const double varphi_std_dev = 1.0;
  // param.k_Psi = 1.0;
  // param.Psi_std_dev = 0.1;

  auto fft_wrapper_N = fftwWrapper(N);
  auto fft_wrapper_M = fftwWrapper(M);
  
  Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(N, L, varphi_std_dev, k_ast, 0);
  Eigen::VectorXd varphi = generate_gaussian_random_field(N, L, P_f);
  
  // Eigen::VectorXd varphi(N*N*N);
  // for(int a = 0; a < N; ++a){
  //   for(int b = 0; b < N; ++b){
  //     for(int c = 0; c < N; ++c){
  // 	varphi(IDX_OF(N, a, b, c)) = cos(2 * pi * a / N);
  //     }
  //   }
  // }
  
  Eigen::VectorXd varphi_fft = fft_wrapper_N.execute_d2z(varphi);
  
  //std::cout << "varphi_fft slice = " << varphi_fft(seqN(N*(N/2+1)*1 + (N/2+1)*0 + 0, 16)).transpose() << '\n';

  Eigen::VectorXd reduced_varphi_fft(2 * M * M * (M / 2 + 1));
  for(int a = 0; a < M; ++a){
    for(int b = 0; b < M; ++b){
      for(int c = 0; c <= M/2; ++c){
	int a_N = (a<M/2) ? a : (N-(M-a));
	int b_N = (b<M/2) ? b : (N-(M-b));
	int c_N = c;
	
	int idx_M = M*(M/2+1)*a + (M/2+1)*b + c;
	int idx_N = N*(N/2+1)*a_N + (N/2+1)*b_N + c_N;
	reduced_varphi_fft(2 * idx_M) = varphi_fft(2 * idx_N) / pow(scale_diff, 3);
	reduced_varphi_fft(2 * idx_M + 1) = varphi_fft(2 * idx_N + 1) / pow(scale_diff, 3);

	// if(abs(varphi_fft(2 * idx_N)) > 1e-6 || abs(varphi_fft(2 * idx_N + 1)) > 1e-6){
	//   std::cout << "found mode: " << a << ',' << b <<  ',' << c << '\n';
	// }
      }
    }
  }
  std::cout << '\n';

  // std::cout << "full scan\n\n";
  // for(int a = 0; a < N; ++a){
  //   for(int b = 0; b < N; ++b){
  //     for(int c = 0; c <= N/2; ++c){
  // 	int idx_N = N*(N/2+1)*a + (N/2+1)*b + c;
  // 	if(abs(varphi_fft(2 * idx_N)) > 1e-6 || abs(varphi_fft(2 * idx_N + 1)) > 1e-6){
  // 	  std::cout << "found mode: " << a << ',' << b <<  ',' << c << '\n';
  // 	}
  //     }
  //   }
  // }
  

  // std::cout << "varphi_fft = " << varphi_fft.head(16).transpose() << "\n";
  // std::cout << "reduced_varphi_fft = " << reduced_varphi_fft.head(16).transpose() << "\n\n";
  Eigen::VectorXd reduced_varphi = fft_wrapper_M.execute_z2d(reduced_varphi_fft) / pow(M, 3);  
  {
    Eigen::VectorXd reduced_varphi_2(M*M*M);

    for(int a = 0; a < M; ++a){
      for(int b = 0; b < M; ++b){
	for(int c = 0; c < M; ++c){
	  reduced_varphi_2(IDX_OF(M, a, b, c)) = varphi(IDX_OF(N, scale_diff*a, scale_diff*b, scale_diff*c));
	}
      }
    } 

    std::cout << "reduced_varphi_2 = " << reduced_varphi_2.head(16).transpose() << '\n';
    std::cout << "reduced_varphi = " << reduced_varphi.head(16).transpose() << '\n';
    std::cout << "diff = " << (reduced_varphi - reduced_varphi_2).norm() / reduced_varphi_2.norm() << '\n';
    std::cout << "reduced_varphi_2.norm() = " << reduced_varphi_2.norm() << '\n';
    std::cout << "reduced_varphi.norm() = " << reduced_varphi.norm() << '\n';
    std::cout << '\n';
  }

  {
    Eigen::VectorXd copied_varphi(N*N*N);
    Eigen::VectorXd interpolated_varphi(N*N*N);

    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  //copied_varphi(IDX_OF(N, a, b, c)) = reduced_varphi(IDX_OF(M, a/3, b/3, c/3));
	  copied_varphi(IDX_OF(N, (a+N-1)%N, (b+N-1)%N, (c+N-1)%N)) = reduced_varphi(IDX_OF(M, a/scale_diff, b/scale_diff, c/scale_diff));

	  // Trilinear interpolation
	  double val_000 = reduced_varphi(IDX_OF(M, a/scale_diff, b/scale_diff, c/scale_diff));
	  double val_001 = reduced_varphi(IDX_OF(M, a/scale_diff, b/scale_diff, (c/scale_diff+1)%M));
	  double val_010 = reduced_varphi(IDX_OF(M, a/scale_diff, (b/scale_diff+1)%M, c/scale_diff));
	  double val_011 = reduced_varphi(IDX_OF(M, a/scale_diff, (b/scale_diff+1)%M, (c/scale_diff+1)%M));
	  double val_100 = reduced_varphi(IDX_OF(M, (a/scale_diff+1)%M, b/scale_diff, c/scale_diff));
	  double val_101 = reduced_varphi(IDX_OF(M, (a/scale_diff+1)%M, b/scale_diff, (c/scale_diff+1)%M));
	  double val_110 = reduced_varphi(IDX_OF(M, (a/scale_diff+1)%M, (b/scale_diff+1)%M, c/scale_diff));
	  double val_111 = reduced_varphi(IDX_OF(M, (a/scale_diff+1)%M, (b/scale_diff+1)%M, (c/scale_diff+1)%M));

	  double coeff_a = (a - 3*(a/scale_diff)) / double(scale_diff);
	  double coeff_b = (b - 3*(b/scale_diff)) / double(scale_diff);
	  double coeff_c = (c - 3*(c/scale_diff)) / double(scale_diff);
	  
	  double val_00 = (1-coeff_c) * val_000 + coeff_c * val_001;
	  double val_01 = (1-coeff_c) * val_010 + coeff_c * val_011;
	  double val_10 = (1-coeff_c) * val_100 + coeff_c * val_101;
	  double val_11 = (1-coeff_c) * val_110 + coeff_c * val_111;

	  double val_0 = (1-coeff_b) * val_00 + coeff_b * val_01;
	  double val_1 = (1-coeff_b) * val_10 + coeff_b * val_11;
	  
	  double val = (1-coeff_a) * val_0 + coeff_a * val_1;
	  
	  interpolated_varphi(IDX_OF(N, a, b, c)) = val;
	}
      }
    } 
    
    std::cout << "interpolated_varphi = " << interpolated_varphi.head(16).transpose() << '\n';
    std::cout << "copied_varphi = " << copied_varphi.head(16).transpose() << '\n';
    std::cout << "varphi = " << varphi.head(16).transpose() << '\n';
    std::cout << "diff_copied = " << (varphi - copied_varphi).norm() / varphi.norm() << '\n';
    std::cout << "diff_interpolated = " << (varphi - interpolated_varphi).norm() / varphi.norm() << '\n';
    
    std::cout << std::fixed << std::setprecision(9) << std::left;
    std::cout << "interpolated_varphi.norm() = " << interpolated_varphi.norm() << '\n';
    std::cout << "copied_varphi.norm() = " << copied_varphi.norm() << '\n';
    std::cout << "varphi.norm() = " << varphi.norm() << '\n';
    std::cout << '\n';
  }

  
}



/*
void test_01(void)
{
  cufftBatchedPlanWrapper cufft_wrapper_1(param.N);
  thrust::device_vector<double> cuda_fft_result = cufft_wrapper_1.execute(workspace.state);
}

void test_02(void)
{
    size_t size = workspace.state.size();
    Eigen::VectorXd cuda_fft_result_eigen(size);
  
      size_t size = cuda_fft_result.size();
    size = 1024;
    std::cout << "size = " << size << '\n';
    std::cout << "sizeof(size) = " << sizeof(size) << '\n';
    std::cout << "typeid(size) = " << boost::typeindex::type_id_runtime(size) << '\n';
}

void test_03(void)
{
  Eigen::VectorXd out(workspace.state.size());
  cudaMemcpy((void *)out.data(), (const void *)thrust::raw_pointer_cast(workspace.state.data()), workspace.state.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
  copy_vector(workspace.state);
  Eigen::VectorXd out = copy_vector(workspace.state);

  Eigen::VectorXd test(1024);
  Eigen::VectorXd test2(1024);
  copy_vector(test, test2);

  std::cout << "test.data() = " << (void *)test.data() << '\n';
  std::cout << "copied.data() = " << (void *)copied.data() << '\n';
  std::cout << "&test = " << (void *)&test << '\n';
  std::cout << "&copied = " << (void *)&copied << '\n';
    
  Eigen::VectorXd test2(1024);    
  std::cout << "&test2 = " << (void *)&test2 << '\n';
  std::cout << "test2.data() = " << (void *)test2.data() << '\n';
    
  Eigen::VectorXd out(workspace.state.size());
  copy_vector(out, workspace.state);
}

void test_04(void)
{
  Eigen::VectorXd eigen_vec(workspace.state.size());
  Eigen::VectorXd eigen_vec_test(workspace.state.size());

  run_and_measure_time("copy from device 1",
		       [&](){ ALGORITHM_NAMESPACE::copy(workspace.state.begin(), workspace.state.end(), eigen_vec_test.begin()); } );
  run_and_measure_time("copy from device 2",
		       [&](){
			 for(int i = 0; i < 100; ++i)
			   cudaMemcpy((void *)eigen_vec.data(), thrust::raw_pointer_cast(workspace.state.data()), eigen_vec.size() * sizeof(double), cudaMemcpyDeviceToHost);
			 cudaStreamSynchronize(0); });

  std::cout << "norm = " << (eigen_vec - eigen_vec_test).norm() << '\n';

  
  Eigen::VectorXd out(workspace.state.size());
  profile_function(100, [&](){ copy_vector(out, workspace.state); });
}

void test_05(void)
{
  cufftBatchedPlanWrapper cufft_wrapper_1(param.N);
  thrust::device_vector<double> cuda_fft_result = cufft_wrapper_1.execute(workspace.state);
  // std::cout << "size = " << cuda_fft_result.size() << '\n';
  // std::cout << "sizeof(size) = " << sizeof(cuda_fft_result.size()) << '\n';
  // std::cout << "type = " << boost::typeindex::type_id_runtime(cuda_fft_result.size()) << '\n';
  //unsigned long long size = cuda_fft_result.size();

  Eigen::VectorXd cuda_fft_result_eigen(cuda_fft_result.size());
  // Eigen::VectorXd cuda_fft_result_eigen(static_cast<size_t>(cuda_fft_result.size()));
  // Eigen::VectorXd cuda_fft_result_eigen(static_cast<std::ptrdiff_t>(cuda_fft_result.size()));
  // cuda_fft_result_eigen.array() = 0;
  // Eigen::VectorXd cuda_fft_result_eigen = Eigen::VectorXd::Zero(cuda_fft_result.size());
    
  copy_vector(cuda_fft_result_eigen, cuda_fft_result);
  std::cout << "eigen size = " << cuda_fft_result_eigen.size() << '\n';
  // auto cuda_fft_result_eigen = copy_vector(cuda_fft_result);
  // std::cout << "cuda_fft_result.data() = " << (void *)thrust::raw_pointer_cast(cuda_fft_result.data()) << '\n';
    
  Eigen::VectorXd eigen_state(workspace.state.size());
  copy_vector(eigen_state, workspace.state);
  Eigen::VectorXd fftw_fft_result(param.N * param.N * (param.N / 2 + 1) * 2 * 2);
  assert(fftw_fft_result.size() == cuda_fft_result_eigen.size());
  fftw_plan plan1 = fftw_plan_dft_r2c_3d(param.N, param.N, param.N, eigen_state.data(), (fftw_complex *)fftw_fft_result.data(), FFTW_ESTIMATE);
  fftw_plan plan2 = fftw_plan_dft_r2c_3d(param.N, param.N, param.N, eigen_state.data() + param.N * param.N * param.N, ((fftw_complex *)fftw_fft_result.data()) + param.N * param.N * (param.N / 2 + 1), FFTW_ESTIMATE);
  fftw_execute(plan1);
  fftw_execute(plan2);

  profile_function(10, [&](){ cufft_wrapper_1.execute(workspace.state); cudaStreamSynchronize(0); });
  profile_function(10, [&](){ fftw_execute(plan1); fftw_execute(plan2); });
    
  fftw_destroy_plan(plan1);
  fftw_destroy_plan(plan2);
  std::cout << "norm = " << (cuda_fft_result_eigen - fftw_fft_result).norm() << '\n';
  std::cout << "norm of cuda_fft_result_eigen = " << cuda_fft_result_eigen.norm() << '\n';
}

void test_06(void)
{
  cufftPlanWrapper cufft_wrapper(param.N);
  auto fft_result = cufft_wrapper.execute(workspace.state);

  Eigen::VectorXd fft_result_eigen(fft_result.size());
  copy_vector(fft_result_eigen, fft_result);

  // Do the same thing using FFTW
  Eigen::VectorXd eigen_v(workspace.state.size());
  copy_vector(eigen_v, workspace.state);
  Eigen::VectorXd eigen_v_fft(param.N * param.N * (param.N / 2 + 1) * 2);

  fftw_plan plan2 = fftw_plan_dft_r2c_3d(param.N, param.N, param.N, eigen_v.data(), (fftw_complex *)eigen_v_fft.data(), FFTW_ESTIMATE);

  profile_function(10, [&](){
			 fftw_execute(plan2);
		       } );
  fftw_destroy_plan(plan2);

  std::cout << "norm = " << (eigen_v_fft - fft_result_eigen).norm() << '\n';
  std::cout << "norm of eigen_v_fft = " << eigen_v_fft.norm() << '\n';
}

void test_07(void)
{
  Vector rho = Equation::compute_energy_density(workspace, 0);
  Eigen::VectorXd rho_eigen(rho.size());
  ALGORITHM_NAMESPACE::copy(rho.begin(), rho.end(), rho_eigen.begin());
  Eigen::VectorXd rho_spectrum = compute_power_spectrum(param.N, rho_eigen);
  Eigen::VectorXd rho_fourier = compute_fourier(param.N, param.L, rho_eigen);
  
  std::cout << "total = " << rho_eigen.sum() << '\n';
  std::cout << "total_spectrum = " << sqrt(rho_spectrum(0)) << '\n';
  std::cout << "rho_fourier_zero_mode = " << rho_fourier(0) << " + " << rho_fourier(1) << 'i' << '\n';
}

void test_08(void)
{
  int NN = 16;
  Eigen::VectorXd eigen_vector_1(NN);
  Eigen::VectorXd eigen_vector_2(NN);
  thrust::device_vector<double> thrust_vector(NN);
  eigen_vector_1.array() = 1.0;
  thrust::copy(eigen_vector_1.begin(), eigen_vector_1.end(), thrust_vector.begin());
  thrust::copy(thrust_vector.begin(), thrust_vector.end(), eigen_vector_2.begin());
  std::cout << "eigen_vector_2 = " << eigen_vector_2 << '\n';
}
*/

/*
#define GRID_SIZE 256

struct TestParam {
  long long int N = GRID_SIZE;
  double L = GRID_SIZE * 10.0;
  double m = 1.0;
  double lambda = 0.1;
  double k_ast = 0.1;
  double k_Psi = 0.03;
  double varphi_std_dev = 1.0;
  double Psi_std_dev = 0.15;
  double a1 = 1;
  double H1 = 0;
  double t1 = 0;
  double t_start = 0;
  double t_end = 1000; //100 * 2 * std::numbers::pi;
  double t_interval = 5;
  double f = 0;
  double dt_f = 1;
};


void test_09(void)
{
  const std::string dir = "../../Data/Run_27/";
  prepare_directory_for_output(dir);
  
  TestParam param;
  print_param(param);
  save_param_for_Mathematica(param, dir);

  typedef CudaLambdaEquationInFRW Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
  typedef typename Equation::Vector Vector;

  Workspace workspace(param, unperturbed_grf);
  Equation eqn(workspace);
  const_interval_observer observer(dir, param, eqn);
}
*/
