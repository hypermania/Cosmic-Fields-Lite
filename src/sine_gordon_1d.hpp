/*
  Tools related to 1D Sine-Gordon fields.
*/
#ifndef SINE_GORDON_1D_HPP
#define SINE_GORDON_1D_HPP

#include "Eigen/Dense"
// #include "random_field.hpp"
// #include "dispatcher.hpp"
// #include "workspace.hpp"
// #include "fdm3d.hpp"

struct SineGordonParam {
  long long int N;
  double L;
  double v;
};

inline auto sg_kink =
  [](const auto param) {
    const long long int lattice_size = param.N * param.N * param.N;
    
    Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);

    // The code is CPU only
    auto &state = workspace.state;
    state = generate_gaussian_random_sp_field(param.N, param.L, P_f);

    workspace.Psi.resize(lattice_size);
    workspace.Psi.array() = 0;
  };

/*! 
  \brief The SineGordon1D equation, \f$ \ddot{\varphi} - \nabla^2 \varphi + m^2 \vec{A} = 0 \f$.
*/
struct SineGordon1DEquation {
  typedef Eigen::ArrayXd State;
  typedef Eigen::ArrayXd Vector;
  typedef SineGordonParam Param;
  // typedef WorkspaceGeneric<Vector, State> Workspace;
  // Workspace &workspace;
  Param param;
  
  // SineGordon1DEquation(Workspace &workspace_) : workspace(workspace_) {}
  SineGordon1DEquation(Param param_) : param(param_) {}

  // TODO
  void operator()(const State &x, State &dxdt, const double) {
    const long long int N = param.N;
    const double L = param.L;
    auto u = x.segment(0, N);
    auto dudt = x.segment(N, N);
    dxdt.segment(0, N) = dudt;
    dxdt.segment(N, N) = -sin(u);
  }

  static Vector compute_energy_density(Workspace &workspace, const double t);
  
  static Vector compute_momentum_density(Workspace &workspace, const double t);
};

#endif
