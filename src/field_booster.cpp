#include "field_booster.hpp"

#include <array>
#include <deque>
#include <boost/math/interpolators/quintic_hermite.hpp>
#include <boost/math/interpolators/cubic_hermite.hpp>

#include "workspace.hpp"
#include "equations.hpp"
#include "fdm3d.hpp"
#include "utility.hpp"
#include "io.hpp"
#include "param.hpp"
#include "initializer.hpp"
#include "random_field.hpp"
//#include "utility.hpp"

#include "proca.hpp"
#include "sp.hpp"
#include "sine_gordon_1d.hpp"


struct KGParam {
  long long int N;
  double L;
  double m;
};

void scan_and_set_klein_gordon(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, double t, const double delta_t, Eigen::VectorXd &state_new)
{
  using namespace boost::numeric::odeint;
  using namespace boost::math::interpolators;
      
  auto empty_initializer = [&](const auto param, auto &workspace) {};
  auto interpolant_at_pos =
    [N](const double t_0, const double t_1,
	const Eigen::VectorXd &state_0, const Eigen::VectorXd &state_1,
	const Eigen::VectorXd &dt_state_0, const Eigen::VectorXd &dt_state_1,
	const int a, const int b, const int c) {
      const int idx = IDX_OF(N, a, b, c);
      quintic_hermite<std::array<double, 2>>
	interpolant(std::array<double, 2>({t_0, t_1}),
		    std::array<double, 2>({state_0(idx), state_1(idx)}),
		    std::array<double, 2>({dt_state_0(idx), dt_state_1(idx)}),
		    std::array<double, 2>({dt_state_0(N*N*N + idx), dt_state_1(N*N*N + idx)}) );
      return interpolant;
    };

  typedef KleinGordonEquation Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
      
  KGParam param = KGParam({N, L, m});
  Workspace workspace(param, empty_initializer);
  Equation eqn(workspace);
  auto stepper = runge_kutta4<State, double, State, double>();

  const double h = L / N;
  const long long int state_size = state_init.size();

  const double t_max = tau.maxCoeff();
  const double t_min = tau.minCoeff();
  std::cout << "t_max = " << t_max << '\n';
  std::cout << "t_min = " << t_min << '\n';
      
  Eigen::VectorXd state_last(state_size);
  Eigen::VectorXd dt_state_last(state_size);
  Eigen::VectorXd state_cur(state_size);
  Eigen::VectorXd dt_state_cur(state_size);

  // Initialization
  state_cur = state_init;
  state_last = state_init;
  eqn(state_last, dt_state_last, t);

  // Loop in one direction
  while(t_min < t && t < t_max) {
    std::cout << "t = " << t << '\n';
    stepper.do_step(eqn, state_cur, t, delta_t);
    eqn(state_cur, dt_state_cur, t);
	
    // Set new initial conditions by interpolation
    const double t0 = std::min(t, t + delta_t);
    const double t1 = std::max(t, t + delta_t);
    const Eigen::VectorXd &state0 = (delta_t > 0) ? state_last : state_cur;
    const Eigen::VectorXd &state1 = (delta_t > 0) ? state_cur : state_last;
    const Eigen::VectorXd &dt_state0 = (delta_t > 0) ? dt_state_last : dt_state_cur;
    const Eigen::VectorXd &dt_state1 = (delta_t > 0) ? dt_state_cur : dt_state_last;
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  const int idx = IDX_OF(N, a, b, c);
	  const double t_eval = tau(idx);
	  if(t0 <= t_eval && t_eval <= t1) {
	    auto center_interpolant = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, b, c);
	    
	    const double delta_varphi_x = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, (a+1)%N, b, c)(t_eval) - interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, (a+N-1)%N, b, c)(t_eval);
	    const double delta_varphi_y = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, (b+1)%N, c)(t_eval) - interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, (b+N-1)%N, c)(t_eval);
	    const double delta_varphi_z = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, b, (c+1)%N)(t_eval) - interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, b, (c+N-1)%N)(t_eval);

	    const double delta_tau_x = tau(IDX_OF(N, (a+1)%N, b, c)) - tau(IDX_OF(N, (a+N-1)%N, b, c));
	    const double delta_tau_y = tau(IDX_OF(N, a, (b+1)%N, c)) - tau(IDX_OF(N, a, (b+N-1)%N, c));
	    const double delta_tau_z = tau(IDX_OF(N, a, b, (c+1)%N)) - tau(IDX_OF(N, a, b, (c+N-1)%N));
	    
	    const double varphi_new = center_interpolant(t_eval);
	    const double dt_varphi_new = center_interpolant.prime(t_eval)
	      + (delta_varphi_x * delta_tau_x + delta_varphi_y * delta_tau_y + delta_varphi_z * delta_tau_z) / (4 * h * h);
	    state_new(idx) = varphi_new;
	    state_new(N*N*N + idx) = dt_varphi_new;
	  }
	}
      }
    }

    // Prepare for next time step
    state_last = state_cur;
    dt_state_last.swap(dt_state_cur);
    t += delta_t;
  }
}

Eigen::VectorXd boost_klein_gordon_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, const double abs_delta_t)
{
  Eigen::VectorXd state_new(state_init.size());
  scan_and_set_klein_gordon(N, L, m, tau, state_init, 0, abs_delta_t, state_new);
  scan_and_set_klein_gordon(N, L, m, tau, state_init, 0, -abs_delta_t, state_new);
  return state_new;
}


void scan_and_set_proca(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, Eigen::VectorXd &state, double t, const double delta_t, Eigen::VectorXd &state_new)
{  
  using namespace boost::numeric::odeint;
  using namespace boost::math::interpolators;
      
  auto empty_initializer = [&](const auto param, auto &workspace) {};
  
  const double h = L / N;
  const long long int field_size = N*N*N;
  const long long int state_size = state.size(); // 6 * field_size

  const double t_max = tau.maxCoeff();
  const double t_min = tau.minCoeff();
  std::cout << "t_max = " << t_max << '\n';
  std::cout << "t_min = " << t_min << '\n';

  Eigen::VectorXd state_next(state_size);
  Eigen::VectorXd At(field_size);
  Eigen::VectorXd At_next(field_size);
  Eigen::VectorXd kg_state(2 * field_size);

  // typedef KleinGordonEquation Equation;
  typedef KleinGordonEquation::Workspace KGWorkspace;
  typedef ProcaEquation::Workspace ProcaWorkspace;
  typedef ProcaWorkspace::State State;
  
  KGParam param = KGParam({N, L, m});
  KGWorkspace workspace_kg(param, empty_initializer);
  ProcaWorkspace workspace_proca(param, empty_initializer);
  KleinGordonEquation eqn_kg(workspace_kg);

  // Initialization
  workspace_proca.state.swap(state);
  At = ProcaEquation::compute_At(workspace_proca, 0);
  workspace_proca.state.swap(state);
  
  // Loop in one direction
  while(t_min < t && t < t_max) {
    std::cout << "t = " << t << '\n';
    
    // Evolve to set state_next
    {
      auto stepper = runge_kutta4<State, double, State, double>();

      auto evolve_component = [&](const long long int i)->void {
	kg_state.segment(0, field_size) = state.segment(i * field_size, field_size);
	kg_state.segment(field_size, field_size) = state.segment((3+i) * field_size, field_size);
	stepper.do_step(eqn_kg, kg_state, t, delta_t);
	state_next.segment(i * field_size, field_size) = kg_state.segment(0, field_size);
	state_next.segment((3+i) * field_size, field_size) = kg_state.segment(field_size, field_size);
      };
      evolve_component(0);
      evolve_component(1);
      evolve_component(2);
    }
    
    // Compute At_next
    {
      workspace_proca.state.swap(state_next);
      At_next = ProcaEquation::compute_At(workspace_proca, 0);
      workspace_proca.state.swap(state_next);
    }
    
    // Set new initial conditions by interpolation
    const double t0 = std::min(t, t + delta_t);
    const double t1 = std::max(t, t + delta_t);
    const Eigen::VectorXd &state0 = (delta_t > 0) ? state : state_next;
    const Eigen::VectorXd &state1 = (delta_t > 0) ? state_next : state;
    const Eigen::VectorXd &At0 = (delta_t > 0) ? At : At_next;
    const Eigen::VectorXd &At1 = (delta_t > 0) ? At_next : At;
    // const Eigen::VectorXd &dt_state0 = (delta_t > 0) ? dt_state_last : dt_state_cur;
    // const Eigen::VectorXd &dt_state1 = (delta_t > 0) ? dt_state_cur : dt_state_last;

    auto field_func_for_lattice = [&](const auto &field) {
      return [&](const int a, const int b, const int c){
	return field(IDX_OF(N, a, b, c));
      };
    };
    
    auto dot_product = [&](const std::array<double, 3> &v1, const std::array<double, 3> &v2)->double {
      return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    };
    
    auto cubic_interpolant =
      [&](const double t_eval,
	  const auto &f0, const auto &f1,
	  const auto &dt_f0, const auto &dt_f1,
	  const int a, const int b, const int c) {
	const int idx = IDX_OF(N, a, b, c);
	cubic_hermite<std::array<double, 2>>
	  interpolant(std::array<double, 2>({t0, t1}),
		      std::array<double, 2>({f0(idx), f1(idx)}),
		      std::array<double, 2>({dt_f0(idx), dt_f1(idx)}) );
	return std::array<double, 2>({interpolant(t_eval), interpolant.prime(t_eval)});
      };
    
    // auto cubic_interpolant_prime =
    //   [&](const auto &f0, const auto &f1,
    // 	  const auto &dt_f0, const auto &dt_f1,
    // 	  const int a, const int b, const int c) {
    // 	const int idx = IDX_OF(N, a, b, c);
    // 	const double t_eval = tau(idx);
    // 	cubic_hermite<std::array<double, 2>>
    // 	  interpolant(std::array<double, 2>({t0, t1}),
    // 		      std::array<double, 2>({f0(idx), f1(idx)}),
    // 		      std::array<double, 2>({dt_f0(idx), dt_f1(idx)}) );
    // 	return interpolant.prime(t_eval);
    //   };
    
    auto Ai_interpolant = [&](const long long int i) {
      return [&, i](const double t_eval, const int a, const int b, const int c) {
	return cubic_interpolant(t_eval,
				 state0.segment(i * field_size, field_size),
				 state1.segment(i * field_size, field_size),
				 state0.segment((3+i) * field_size, field_size),
				 state1.segment((3+i) * field_size, field_size),
				 a, b, c)[0];
      };
    };
    
    auto dt_Ai_interpolant = [&](const long long int i) {
      return [&, i](const double t_eval, const int a, const int b, const int c) {
	return cubic_interpolant(t_eval,
				 state0.segment(i * field_size, field_size),
				 state1.segment(i * field_size, field_size),
				 state0.segment((3+i) * field_size, field_size),
				 state1.segment((3+i) * field_size, field_size),
				 a, b, c)[1];
      };
    };

    
    auto A1_interpolant = Ai_interpolant(0);
    auto A2_interpolant = Ai_interpolant(1);
    auto A3_interpolant = Ai_interpolant(2);
    auto dt_A1_interpolant = dt_Ai_interpolant(0);
    auto dt_A2_interpolant = dt_Ai_interpolant(1);
    auto dt_A3_interpolant = dt_Ai_interpolant(2);
    
    auto tau_func = field_func_for_lattice(tau);
    
    auto A1_new = state_new.segment(0 * field_size, field_size);
    auto A2_new = state_new.segment(1 * field_size, field_size);
    auto A3_new = state_new.segment(2 * field_size, field_size);
    auto dt_A1_new = state_new.segment(3 * field_size, field_size);
    auto dt_A2_new = state_new.segment(4 * field_size, field_size);
    auto dt_A3_new = state_new.segment(5 * field_size, field_size);
  
    auto gradient_at_pos = [&](auto field_func, const int a, const int b, const int c) {
      const double grad_x = (field_func((a+1)%N, b, c) - field_func((a+N-1)%N, b, c)) / (2 * h);
      const double grad_y = (field_func(a, (b+1)%N, c) - field_func(a, (b+N-1)%N, c)) / (2 * h);
      const double grad_z = (field_func(a, b, (c+1)%N) - field_func(a, b, (c+N-1)%N)) / (2 * h);
      return std::array<double, 3>({grad_x, grad_y, grad_z});
    };

    auto gradient_at_pos_time = [&](auto field_func, const double t_eval, const int a, const int b, const int c) {
      const double grad_x = (field_func(t_eval, (a+1)%N, b, c) - field_func(t_eval, (a+N-1)%N, b, c)) / (2 * h);
      const double grad_y = (field_func(t_eval, a, (b+1)%N, c) - field_func(t_eval, a, (b+N-1)%N, c)) / (2 * h);
      const double grad_z = (field_func(t_eval, a, b, (c+1)%N) - field_func(t_eval, a, b, (c+N-1)%N)) / (2 * h);
      return std::array<double, 3>({grad_x, grad_y, grad_z});
    };

    // const double A1_interpolated = A1_interpolant(t_eval, a, b, c);
    // const double A2_interpolated = A2_interpolant(t_eval, a, b, c);
    // const double A3_interpolated = A3_interpolant(t_eval, a, b, c);
    // const double dt_A1_interpolated = dt_A1_interpolant(t_eval, a, b, c);
    // const double dt_A2_interpolated = dt_A2_interpolant(t_eval, a, b, c);
    // const double dt_A3_interpolated = dt_A3_interpolant(t_eval, a, b, c);

    auto A10 = state0.segment(0 * field_size, field_size);
    auto A20 = state0.segment(1 * field_size, field_size);
    auto A30 = state0.segment(2 * field_size, field_size);
    auto A11 = state1.segment(0 * field_size, field_size);
    auto A21 = state1.segment(1 * field_size, field_size);
    auto A31 = state1.segment(2 * field_size, field_size);
    auto At_interpolant = [&](const double t_eval, const int a, const int b, const int c) {
      const double div_A_0 =
	(A10(IDX_OF(N, (a+1)%N, b, c)) - A10(IDX_OF(N, (a+N-1)%N, b, c))) / (2 * h)
	+ (A20(IDX_OF(N, a, (b+1)%N, c)) - A20(IDX_OF(N, a, (b+N-1)%N, c))) / (2 * h)
	+ (A30(IDX_OF(N, a, b, (c+1)%N)) - A30(IDX_OF(N, a, b, (c+N-1)%N))) / (2 * h);
      const double div_A_1 =
	(A11(IDX_OF(N, (a+1)%N, b, c)) - A11(IDX_OF(N, (a+N-1)%N, b, c))) / (2 * h)
	+ (A21(IDX_OF(N, a, (b+1)%N, c)) - A21(IDX_OF(N, a, (b+N-1)%N, c))) / (2 * h)
	+ (A31(IDX_OF(N, a, b, (c+1)%N)) - A31(IDX_OF(N, a, b, (c+N-1)%N))) / (2 * h);

      const int idx = IDX_OF(N, a, b, c);
      // const double t_eval = tau(idx);
      cubic_hermite<std::array<double, 2>>
	interpolant(std::array<double, 2>({t0, t1}),
		    std::array<double, 2>({At0(idx), At1(idx)}),
		    std::array<double, 2>({div_A_0, div_A_1}) );
      return interpolant(t_eval);
    };

    
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  const int idx = IDX_OF(N, a, b, c);
	  const double t_eval = tau(idx);
	  
	  if(t0 <= t_eval && t_eval <= t1) {

	    auto grad_tau = gradient_at_pos(tau_func, a, b, c);
	    auto grad_A1 = gradient_at_pos_time(A1_interpolant, t_eval, a, b, c);
	    auto grad_A2 = gradient_at_pos_time(A2_interpolant, t_eval, a, b, c);
	    auto grad_A3 = gradient_at_pos_time(A3_interpolant, t_eval, a, b, c);
	    auto div_A = grad_A1[0] + grad_A2[1] + grad_A3[2];
	    const double At_interpolated = At_interpolant(t_eval,a,b,c);
	    A1_new(idx) = A1_interpolant(t_eval,a,b,c) + grad_tau[0] * At_interpolated;
	    A2_new(idx) = A2_interpolant(t_eval,a,b,c) + grad_tau[1] * At_interpolated;
	    A3_new(idx) = A3_interpolant(t_eval,a,b,c) + grad_tau[2] * At_interpolated;
	    dt_A1_new(idx) = dt_A1_interpolant(t_eval,a,b,c) + dot_product(grad_tau, grad_A1) + grad_tau[0] * div_A;
	    dt_A2_new(idx) = dt_A2_interpolant(t_eval,a,b,c) + dot_product(grad_tau, grad_A2) + grad_tau[1] * div_A;
	    dt_A3_new(idx) = dt_A3_interpolant(t_eval,a,b,c) + dot_product(grad_tau, grad_A3) + grad_tau[2] * div_A;
		    
	  }
	}
      }
    }

    // Prepare for next time step
    state.swap(state_next);
    At.swap(At_next);
    t += delta_t;
  }
}

Eigen::VectorXd boost_proca_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, Eigen::VectorXd &state_init, const double abs_delta_t, const std::string save_path)
{
  const long long int lattice_size = N*N*N;  
  Eigen::VectorXd state_new(6 * lattice_size);
  
  write_to_file(state_init, save_path);
  // Eigen::VectorXd temp = state_init;
  scan_and_set_proca(N, L, m, tau, state_init, 0, abs_delta_t, state_new);
  Eigen::VectorXd().swap(state_init);
  // state_init.swap(temp);
  state_init = load_VectorXd_from_file(save_path);
  scan_and_set_proca(N, L, m, tau, state_init, 0, -abs_delta_t, state_new);
  
  return state_new;
}

Eigen::ArrayXcd boost_sp_field(const long long int N, const double L, const double m, const Eigen::ArrayXd &tau, const Eigen::ArrayXcd &state_init)
{
  const long long int lattice_size = N*N*N;
  Eigen::ArrayXcd state_new(state_init.size());
  
  Eigen::ArrayXcd rotation(lattice_size);
  rotation = exp(std::complex<double>(0, -1) * m * tau);
  
  state_new.segment(0 * lattice_size, lattice_size) = rotation * state_init.segment(0 * lattice_size, lattice_size);
  state_new.segment(1 * lattice_size, lattice_size) = rotation * state_init.segment(1 * lattice_size, lattice_size);
  state_new.segment(2 * lattice_size, lattice_size) = rotation * state_init.segment(2 * lattice_size, lattice_size);

  return state_new;
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


void generate_ic_sg(void)
{
  const std::string dir = "/media/hypermania/Drive_001/FreeStreamingULDM/sg_IC/";
  prepare_directory_for_output(dir);
  
  using namespace Eigen;
  using namespace std::numbers;
  using namespace boost::numeric::odeint;
  const double L = 40;
  const double x0 = 0.25 * L;
  const double x1 = 0.75 * L;
  const long long int N = static_cast<long long int>(L / 0.01);
  const double omega = 0.2;
  
  SineGordonParam param {
    .N = N,
    .L = L,
    .v = 0
  };
  print_param(param);
  save_param_for_Mathematica(param, dir);

  
  typedef SineGordon1DEquation Equation;
  typedef SineGordon1DEquation::State State;
  Equation eqn(param);
  Equation::State state(2 * N);

  // Initialize breather solutions with frequency omega at x0 and x1
  Equation::Vector xCoords = Eigen::ArrayXd::LinSpaced(N, 0, (L * (N - 1))  / N);
  state(seqN(0, N)) = 0;
  state(seqN(N, N)) = ((4)*((pow((1)+((-1)*(pow(omega,2))),0.500000000000000000000000000000))*(1/cosh(((xCoords)+((-1)*(x0)))*(pow((1)+((-1)*(pow(omega,2))),0.500000000000000000000000000000))))))+((4)*((pow((1)+((-1)*(pow(omega,2))),0.500000000000000000000000000000))*(1/cosh(((xCoords)+((-1)*(x1)))*(pow((1)+((-1)*(pow(omega,2))),0.500000000000000000000000000000))))));

  // Function to coarse grain a field over the periodic grid.
  // At index idx, averages the field over [idx - window, idx + window].
  auto periodic_smoothing = [](const long long int window, const Equation::Vector &field)->Equation::Vector {
    using namespace Eigen;
    Equation::Vector extended_field(3 * field.size());

    extended_field(seqN(0 * field.size(), field.size())) = field;
    extended_field(seqN(1 * field.size(), field.size())) = field;
    extended_field(seqN(2 * field.size(), field.size())) = field;

    Equation::Vector smoothed_field(field.size());
    for(long long int idx = 0; idx < field.size(); ++idx) {
      smoothed_field(idx) = extended_field(seqN(field.size() + idx - window, 2 * window)).mean();
    }
    return smoothed_field;
  };

  const long long int window = 200;
  auto rho = eqn.compute_energy_density(state, 0);
  auto q = eqn.compute_momentum_density(state, 0);
  auto p = eqn.compute_pressure(state, 0);
  write_to_file(state, dir + "state.dat");
  write_to_file(rho, dir + "rho.dat");
  write_to_file(q, dir + "q.dat");
  write_to_file(p, dir + "p.dat");
  write_to_file(periodic_smoothing(window, rho), dir + "rho_smoothed.dat");
  write_to_file(periodic_smoothing(window, q), dir + "q_smoothed.dat");
  write_to_file(periodic_smoothing(window, p), dir + "p_smoothed.dat");

  typedef SineGordon1DBooster Booster;
  Eigen::ArrayXd tau = 2 * cos(2 * pi * xCoords / L) * (L / (2 * pi));
  Booster booster(param, tau);
  
  auto stepper = runge_kutta4_classic<State, double, State, double>();
  // auto stepper = make_controlled(1e-9, 1e-9, runge_kutta_fehlberg78<State, double, State, double>());
    
  int num_steps = integrate_const(stepper, booster, state, 0.0, 1.0, 0.0001); //, observer);

  rho = eqn.compute_energy_density(state, 0);
  q = eqn.compute_momentum_density(state, 0);
  p = eqn.compute_pressure(state, 0);
  write_to_file(state, dir + "state_boosted.dat");
  write_to_file(rho, dir + "rho_boosted.dat");
  write_to_file(q, dir + "q_boosted.dat");
  write_to_file(p, dir + "p_boosted.dat");
  write_to_file(periodic_smoothing(window, rho), dir + "rho_boosted_smoothed.dat");
  write_to_file(periodic_smoothing(window, q), dir + "q_boosted_smoothed.dat");
  write_to_file(periodic_smoothing(window, p), dir + "p_boosted_smoothed.dat");

  write_to_file(tau, dir + "tau.dat");
  write_to_file(xCoords, dir + "x_coords.dat");
}
