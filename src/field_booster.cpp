#include "field_booster.hpp"

#include <array>
#include <deque>
#include <boost/math/interpolators/quintic_hermite.hpp>

#include "workspace.hpp"
#include "equations.hpp"
#include "fdm3d.hpp"
#include "utility.hpp"


void add_phase_to_state(Eigen::VectorXd &state, const Eigen::VectorXd &phase)
{
  auto field_size = state.size() / 2;
  Eigen::ArrayXd f = phase.array().cos() * state.head(field_size).array() -
    phase.array().sin() * state.tail(field_size).array();
  Eigen::ArrayXd dtf = phase.array().sin() * state.head(field_size).array() +
    phase.array().cos() * state.tail(field_size).array();
  state.head(field_size) = f;
  state.tail(field_size) = dtf;
}

struct KGParam {
  long long int N;
  double L;
  double m;
};

void boost_klein_gordon_field_old(Eigen::VectorXd &varphi, Eigen::VectorXd &dt_varphi, const Eigen::VectorXd &theta,
				  const long long int N, const double L, const double m)
{
  using namespace boost::numeric::odeint;
  using namespace boost::math::interpolators;
  typedef KleinGordonEquation Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;

  const double h = L / N;
  
  auto make_state = [](const Eigen::VectorXd &f, const Eigen::VectorXd &dt_f) {
		      Eigen::VectorXd state(f.size() + dt_f.size());
		      state.head(f.size()) = f;
		      state.tail(dt_f.size()) = dt_f;
		      return state;
		    };
  
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

  
  KGParam param = KGParam({N, L, m});
  Workspace workspace(param, empty_initializer);
  Equation eqn(workspace);
  auto stepper = runge_kutta4<State, double, State, double>();

  const double delta_t = 0.01 / m;
  const double t_max = - theta.minCoeff() / m;
  const double t_min = - theta.maxCoeff() / m;

  std::cout << "t_max = " << t_max << '\n';
  std::cout << "t_min = " << t_min << '\n';

  Eigen::VectorXd varphi_new(varphi.size());
  Eigen::VectorXd dt_varphi_new(dt_varphi.size());

  const long long int state_size = varphi.size() + dt_varphi.size();
  std::deque<Eigen::VectorXd> state_buffer;
  std::deque<Eigen::VectorXd> dt_state_buffer;
  
  // Solve the equation forward in time and set new initial conditions by interpolation
  workspace.state = make_state(varphi, dt_varphi);
  state_buffer.push_back(Eigen::VectorXd(workspace.state));
  dt_state_buffer.push_back(Eigen::VectorXd(state_size));
  eqn(state_buffer.back(), dt_state_buffer.back(), 0.0);
  
  double t = 0;
  while(t < t_max) {
    std::cout << "t = " << t << '\n';
    stepper.do_step(eqn, workspace.state, 0.0, delta_t);
    state_buffer.push_back(Eigen::VectorXd(workspace.state));
    dt_state_buffer.push_back(Eigen::VectorXd(state_size));
    eqn(state_buffer.back(), dt_state_buffer.back(), 0.0);

    // Set new initial conditions by interpolation
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  const int idx = IDX_OF(N, a, b, c);
	  const double t_eval = -theta(idx) / m;
	  if(t_eval >= t && t_eval <= t + delta_t) {
	    auto center_interpolant = interpolant_at_pos(t, t + delta_t,
							 state_buffer[0], state_buffer[1],
							 dt_state_buffer[0], dt_state_buffer[1],
							 a, b, c);
	    
	    const double delta_varphi_x = interpolant_at_pos(t, t + delta_t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], (a+1)%N, b, c)(t_eval) - interpolant_at_pos(t, t + delta_t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], (a+N-1)%N, b, c)(t_eval);
	    const double delta_varphi_y = interpolant_at_pos(t, t + delta_t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, (b+1)%N, c)(t_eval) - interpolant_at_pos(t, t + delta_t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, (b+N-1)%N, c)(t_eval);
	    const double delta_varphi_z = interpolant_at_pos(t, t + delta_t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, b, (c+1)%N)(t_eval) - interpolant_at_pos(t, t + delta_t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, b, (c+N-1)%N)(t_eval);

	    const double delta_theta_x = theta(IDX_OF(N, (a+1)%N, b, c)) - theta(IDX_OF(N, (a+N-1)%N, b, c));
	    const double delta_theta_y = theta(IDX_OF(N, a, (b+1)%N, c)) - theta(IDX_OF(N, a, (b+N-1)%N, c));
	    const double delta_theta_z = theta(IDX_OF(N, a, b, (c+1)%N)) - theta(IDX_OF(N, a, b, (c+N-1)%N));
	    
	    varphi_new(idx) = center_interpolant(t_eval);
	    dt_varphi_new(idx) = center_interpolant.prime(t_eval)
	      - (delta_varphi_x * delta_theta_x + delta_varphi_y * delta_theta_y + delta_varphi_z * delta_theta_z) / (4 * h * h * m);
	  } 
	}
      }
    }
    // Next time step
    state_buffer.pop_front();
    dt_state_buffer.pop_front();
    t += delta_t;
  }
  state_buffer.clear();
  dt_state_buffer.clear();
  
  std::cout << "point 2\n";
  
  // Solve the equation backward in time and set new initial conditions by interpolation
  workspace.state = make_state(varphi, dt_varphi);
  state_buffer.push_back(Eigen::VectorXd(workspace.state));
  dt_state_buffer.push_back(Eigen::VectorXd(state_size));
  eqn(state_buffer.back(), dt_state_buffer.back(), 0.0);
  
  t = 0;
  while(t > t_min) {
    std::cout << "t = " << t << '\n';
    stepper.do_step(eqn, workspace.state, 0.0, -delta_t);
    state_buffer.push_front(Eigen::VectorXd(workspace.state));
    dt_state_buffer.push_front(Eigen::VectorXd(state_size));
    eqn(state_buffer.front(), dt_state_buffer.front(), 0.0);

    // Set new initial conditions by interpolation
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  const int idx = IDX_OF(N, a, b, c);
	  const double t_eval = -theta(idx) / m;
	  if(t_eval >= t - delta_t && t_eval <= t) {
	    auto center_interpolant = interpolant_at_pos(t - delta_t, t,
							 state_buffer[0], state_buffer[1],
							 dt_state_buffer[0], dt_state_buffer[1],
							 a, b, c);
	    
	    const double delta_varphi_x = interpolant_at_pos(t - delta_t, t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], (a+1)%N, b, c)(t_eval) - interpolant_at_pos(t - delta_t, t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], (a+N-1)%N, b, c)(t_eval);
	    const double delta_varphi_y = interpolant_at_pos(t - delta_t, t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, (b+1)%N, c)(t_eval) - interpolant_at_pos(t - delta_t, t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, (b+N-1)%N, c)(t_eval);
	    const double delta_varphi_z = interpolant_at_pos(t - delta_t, t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, b, (c+1)%N)(t_eval) - interpolant_at_pos(t - delta_t, t, state_buffer[0], state_buffer[1], dt_state_buffer[0], dt_state_buffer[1], a, b, (c+N-1)%N)(t_eval);

	    const double delta_theta_x = theta(IDX_OF(N, (a+1)%N, b, c)) - theta(IDX_OF(N, (a+N-1)%N, b, c));
	    const double delta_theta_y = theta(IDX_OF(N, a, (b+1)%N, c)) - theta(IDX_OF(N, a, (b+N-1)%N, c));
	    const double delta_theta_z = theta(IDX_OF(N, a, b, (c+1)%N)) - theta(IDX_OF(N, a, b, (c+N-1)%N));
	    
	    varphi_new(idx) = center_interpolant(t_eval);
	    dt_varphi_new(idx) = center_interpolant.prime(t_eval)
	      - (delta_varphi_x * delta_theta_x + delta_varphi_y * delta_theta_y + delta_varphi_z * delta_theta_z) / (4 * h * h * m);
	  } 
	}
      }
    }
    // Next time step
    state_buffer.pop_back();
    dt_state_buffer.pop_back();
    t -= delta_t;
  }
  state_buffer.clear();
  dt_state_buffer.clear();

  
  // Save initial conditions
  varphi = varphi_new;
  dt_varphi = dt_varphi_new;
}

void scan_and_set_with_klein_gordon(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, double t, const double delta_t, Eigen::VectorXd &state_new)
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
    for(int a = 0; a < N; ++a){
      for(int b = 0; b < N; ++b){
	for(int c = 0; c < N; ++c){
	  const int idx = IDX_OF(N, a, b, c);
	  const double t_eval = tau(idx);
	  if(t0 <= t_eval && t_eval <= t1) {
	    auto center_interpolant = interpolant_at_pos(t0, t1,
							 state_last, state_cur,
							 dt_state_last, dt_state_cur,
							 a, b, c);
	    
	    const double delta_varphi_x = interpolant_at_pos(t0, t1, state_last, state_cur, dt_state_last, dt_state_cur, (a+1)%N, b, c)(t_eval) - interpolant_at_pos(t0, t1, state_last, state_cur, dt_state_last, dt_state_cur, (a+N-1)%N, b, c)(t_eval);
	    const double delta_varphi_y = interpolant_at_pos(t0, t1, state_last, state_cur, dt_state_last, dt_state_cur, a, (b+1)%N, c)(t_eval) - interpolant_at_pos(t0, t1, state_last, state_cur, dt_state_last, dt_state_cur, a, (b+N-1)%N, c)(t_eval);
	    const double delta_varphi_z = interpolant_at_pos(t0, t1, state_last, state_cur, dt_state_last, dt_state_cur, a, b, (c+1)%N)(t_eval) - interpolant_at_pos(t0, t1, state_last, state_cur, dt_state_last, dt_state_cur, a, b, (c+N-1)%N)(t_eval);

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
  scan_and_set_with_klein_gordon(N, L, m, tau, state_init, 0, abs_delta_t, state_new);
  scan_and_set_with_klein_gordon(N, L, m, tau, state_init, 0, -abs_delta_t, state_new);
  return state_new;
}

void boost_proca_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, double t, const double delta_t, Eigen::VectorXd &state_new)
{
  return;
}

  
// auto evolve_and_set_IC =
//   [N,L,m](const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, double t, const double delta_t, Eigen::VectorXd &state_new)->void {

//   };

