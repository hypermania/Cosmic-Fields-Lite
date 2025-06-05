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

// TODO
void scan_and_set_proca(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, Eigen::VectorXd &state_init, double t, const double delta_t)
{
  // Design decisions:
  // 1. Don't implement an actual operator() for Proca, just use KleinGordonEquation on each A_i component
  // 2. Store the full field A_t, A_i's for two times
  // 3. Compute A_t after evolving A_i's
  // 4. Using KleinGordonEquation don't need a workspace. We can create and destruct the eqn as we go.
  // 5. We will need only one scratch state for the KG stepper to act on.
  // 6. Start with cubic interpolation.
  
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


  const double h = L / N;
  const long long int field_size = N*N*N;
  const long long int state_size = state_init.size(); // 6 * field_size

  const double t_max = tau.maxCoeff();
  const double t_min = tau.minCoeff();
  std::cout << "t_max = " << t_max << '\n';
  std::cout << "t_min = " << t_min << '\n';
      
  Eigen::VectorXd state_next(state_size);
  Eigen::VectorXd kg_state(2 * field_size);

  typedef KleinGordonEquation Equation;
  typedef typename Equation::Workspace Workspace;
  typedef typename Equation::State State;
      
  KGParam param = KGParam({N, L, m});
  Workspace workspace(param, empty_initializer);
  Equation eqn(workspace);

  // Loop in one direction
  while(t_min < t && t < t_max) {
    // Evolve to set state_next
    std::cout << "t = " << t << '\n';
    {
      auto stepper = runge_kutta4<State, double, State, double>();

      auto evolve_component = [&](const long long int i)->void {
	kg_state.segment(0, field_size) = state_init.segment(i * field_size, field_size);
	kg_state.segment(field_size, field_size) = state_init.segment((3+i) * field_size, field_size);
	stepper.do_step(eqn, kg_state, t, delta_t);
	state_next.segment(i * field_size, field_size) = kg_state.segment(0, field_size);
	state_next.segment((3+i) * field_size, field_size) = kg_state.segment(field_size, field_size);
      };
      evolve_component(0);
      evolve_component(1);
      evolve_component(2);
    }
	
    // Set new initial conditions by interpolation
    // const double t0 = std::min(t, t + delta_t);
    // const double t1 = std::max(t, t + delta_t);
    // const Eigen::VectorXd &state0 = (delta_t > 0) ? state_last : state_cur;
    // const Eigen::VectorXd &state1 = (delta_t > 0) ? state_cur : state_last;
    // const Eigen::VectorXd &dt_state0 = (delta_t > 0) ? dt_state_last : dt_state_cur;
    // const Eigen::VectorXd &dt_state1 = (delta_t > 0) ? dt_state_cur : dt_state_last;
    // for(int a = 0; a < N; ++a){
    //   for(int b = 0; b < N; ++b){
    // 	for(int c = 0; c < N; ++c){
    // 	  const int idx = IDX_OF(N, a, b, c);
    // 	  const double t_eval = tau(idx);
    // 	  if(t0 <= t_eval && t_eval <= t1) {
    // 	    auto center_interpolant = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, b, c);
	    
    // 	    const double delta_varphi_x = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, (a+1)%N, b, c)(t_eval) - interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, (a+N-1)%N, b, c)(t_eval);
    // 	    const double delta_varphi_y = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, (b+1)%N, c)(t_eval) - interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, (b+N-1)%N, c)(t_eval);
    // 	    const double delta_varphi_z = interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, b, (c+1)%N)(t_eval) - interpolant_at_pos(t0, t1, state0, state1, dt_state0, dt_state1, a, b, (c+N-1)%N)(t_eval);

    // 	    const double delta_tau_x = tau(IDX_OF(N, (a+1)%N, b, c)) - tau(IDX_OF(N, (a+N-1)%N, b, c));
    // 	    const double delta_tau_y = tau(IDX_OF(N, a, (b+1)%N, c)) - tau(IDX_OF(N, a, (b+N-1)%N, c));
    // 	    const double delta_tau_z = tau(IDX_OF(N, a, b, (c+1)%N)) - tau(IDX_OF(N, a, b, (c+N-1)%N));
	    
    // 	    const double varphi_new = center_interpolant(t_eval);
    // 	    const double dt_varphi_new = center_interpolant.prime(t_eval)
    // 	      + (delta_varphi_x * delta_tau_x + delta_varphi_y * delta_tau_y + delta_varphi_z * delta_tau_z) / (4 * h * h);
    // 	    state_new(idx) = varphi_new;
    // 	    state_new(N*N*N + idx) = dt_varphi_new;
    // 	  }
    // 	}
    //   }
    // }

    // Prepare for next time step
    // state_last = state_cur;
    // dt_state_last.swap(dt_state_cur);
    t += delta_t;
  }
}

// TODO
Eigen::VectorXd boost_proca_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, const double abs_delta_t)
{
  // The current scheme is problematic, because we are missing A_0 contribution to the boost of A_i's
  // To modify this, we need to:
  // Evolve the field by one time step, obtain field's A_i's for t_0 and t_1
  // Compute A_0 at times t_0 and t_1
  // At (\tau(\bx), \bx) such that t_0 <= \tau < t_1, interpolate A_0 and A_i's, and set the boosted field
  Eigen::VectorXd state_new(state_init.size());

  const long long int lattice_size = N*N*N;
  Eigen::VectorXd component_init(2 * lattice_size);
  Eigen::VectorXd component_new(2 * lattice_size);

  component_init.segment(0, lattice_size) = state_init.segment(0, lattice_size);
  component_init.segment(lattice_size, lattice_size) = state_init.segment(3 * lattice_size, lattice_size);
  scan_and_set_klein_gordon(N, L, m, tau, component_init, 0, abs_delta_t, component_new);
  scan_and_set_klein_gordon(N, L, m, tau, component_init, 0, -abs_delta_t, component_new);
  state_new.segment(0, lattice_size) = component_new.segment(0, lattice_size);
  state_new.segment(3 * lattice_size, lattice_size) = component_new.segment(lattice_size, lattice_size);

  component_init.segment(0, lattice_size) = state_init.segment(lattice_size, lattice_size);
  component_init.segment(lattice_size, lattice_size) = state_init.segment(4 * lattice_size, lattice_size);
  scan_and_set_klein_gordon(N, L, m, tau, component_init, 0, abs_delta_t, component_new);
  scan_and_set_klein_gordon(N, L, m, tau, component_init, 0, -abs_delta_t, component_new);
  state_new.segment(lattice_size, lattice_size) = component_new.segment(0, lattice_size);
  state_new.segment(4 * lattice_size, lattice_size) = component_new.segment(lattice_size, lattice_size);

  component_init.segment(0, lattice_size) = state_init.segment(2 * lattice_size, lattice_size);
  component_init.segment(lattice_size, lattice_size) = state_init.segment(5 * lattice_size, lattice_size);
  scan_and_set_klein_gordon(N, L, m, tau, component_init, 0, abs_delta_t, component_new);
  scan_and_set_klein_gordon(N, L, m, tau, component_init, 0, -abs_delta_t, component_new);
  state_new.segment(2 * lattice_size, lattice_size) = component_new.segment(0, lattice_size);
  state_new.segment(5 * lattice_size, lattice_size) = component_new.segment(lattice_size, lattice_size);
  
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
