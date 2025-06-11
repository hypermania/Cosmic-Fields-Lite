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

// inline auto sg_kink =
//   [](const auto param) {
//     const long long int lattice_size = param.N * param.N * param.N;
    
//     Spectrum P_f = power_law_with_cutoff_given_amplitude_3d(param.N, param.L, param.varphi_std_dev, param.k_ast, 0);

//     // The code is CPU only
//     auto &state = workspace.state;
//     state = generate_gaussian_random_sp_field(param.N, param.L, P_f);

//     workspace.Psi.resize(lattice_size);
//     workspace.Psi.array() = 0;
//   };

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
    using namespace Eigen;
    const long long int N = param.N;
    const double L = param.L;
    const double h = L / N;
    auto u = x(seqN(0, N));
    auto dudt = x(seqN(N, N));

    dxdt(seqN(0, N)) = dudt;
    dxdt(seqN(N+2, N-4)) = ((-1)*(sin(x(seqN((2),((-4)+(N)))))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((-1)*(x(seqN((0),((-4)+(N))))))+(((16)*(x(seqN((1),((-4)+(N))))))+(((-30)*(x(seqN((2),((-4)+(N))))))+(((16)*(x(seqN((3),((-4)+(N))))))+((-1)*(x(seqN((4),((-4)+(N))))))))))));
    dxdt(N+1) = ((-1)*(sin(x(1))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((16)*(x(0)))+(((-30)*(x(1)))+(((16)*(x(2)))+(((-1)*(x(3)))+((-1)*(x((-1)+(N))))))))));
    dxdt(N+0) = ((-1)*(sin(x(0))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((-30)*(x(0)))+(((16)*(x(1)))+(((-1)*(x(2)))+(((-1)*(x((-2)+(N))))+((16)*(x((-1)+(N))))))))));
    dxdt(N+N-2) = ((-1)*(sin(x((-2)+(N)))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((-1)*(x(0)))+(((-1)*(x((-4)+(N))))+(((16)*(x((-3)+(N))))+(((-30)*(x((-2)+(N))))+((16)*(x((-1)+(N))))))))));
    dxdt(N+N-1) = ((-1)*(sin(x((-1)+(N)))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((16)*(x(0)))+(((-1)*(x(1)))+(((-1)*(x((-3)+(N))))+(((16)*(x((-2)+(N))))+((-30)*(x((-1)+(N))))))))));

  }

  // static Vector compute_energy_density(Workspace &workspace, const double t);
  
  // static Vector compute_momentum_density(Workspace &workspace, const double t);
};

#endif
