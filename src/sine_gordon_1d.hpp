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

  Vector compute_energy_density(const State &x, const double t) {
    using namespace Eigen;
    const long long int N = param.N;
    const double L = param.L;
    const double h = L / N;
    Vector rho(N);

    rho(seqN(2, N-4)) = (1)+(((-1)*(cos(x(seqN((2),((-4)+(N)))))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow((x(seqN((0),((-4)+(N)))))+(((-8)*(x(seqN((1),((-4)+(N))))))+(((8)*(x(seqN((3),((-4)+(N))))))+((-1)*(x(seqN((4),((-4)+(N)))))))),2))))+((0.500000000000000000000000000000)*(pow(x(seqN(((2)+(N)),((-4)+(N)))),2)))));
    rho(1) = (1)+(((-1)*(cos(x(1))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((-8)*(x(0)))+(((8)*(x(2)))+(((-1)*(x(3)))+(x((-1)+(N))))),2))))+((0.500000000000000000000000000000)*(pow(x((1)+(N)),2)))));
    rho(0) = (1)+(((-1)*(cos(x(0))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((8)*(x(1)))+(((-1)*(x(2)))+((x((-2)+(N)))+((-8)*(x((-1)+(N)))))),2))))+((0.500000000000000000000000000000)*(pow(x(N),2)))));
    rho(N-2) = (1)+(((-1)*(cos(x((-2)+(N)))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((-1)*(x(0)))+((x((-4)+(N)))+(((-8)*(x((-3)+(N))))+((8)*(x((-1)+(N)))))),2))))+((0.500000000000000000000000000000)*(pow(x((-2)+((2)*(N))),2)))));
    rho(N-1) = (1)+(((-1)*(cos(x((-1)+(N)))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((8)*(x(0)))+(((-1)*(x(1)))+((x((-3)+(N)))+((-8)*(x((-2)+(N)))))),2))))+((0.500000000000000000000000000000)*(pow(x((-1)+((2)*(N))),2)))));
    return rho;
  }
  
  Vector compute_momentum_density(const State &x, const double t) {
    using namespace Eigen;
    const long long int N = param.N;
    const double L = param.L;
    const double h = L / N;
    Vector q(N);

    q(seqN(2, N-4)) = (-0.0833333333333333333333333333333)*((pow(h,-1))*(((x(seqN((0),((-4)+(N)))))+(((-8)*(x(seqN((1),((-4)+(N))))))+(((8)*(x(seqN((3),((-4)+(N))))))+((-1)*(x(seqN((4),((-4)+(N)))))))))*(x(seqN(((2)+(N)),((-4)+(N)))))));
    q(1) = (-0.0833333333333333333333333333333)*((pow(h,-1))*((((-8)*(x(0)))+(((8)*(x(2)))+(((-1)*(x(3)))+(x((-1)+(N))))))*(x((1)+(N)))));
    q(0) = (-0.0833333333333333333333333333333)*((pow(h,-1))*((((8)*(x(1)))+(((-1)*(x(2)))+((x((-2)+(N)))+((-8)*(x((-1)+(N)))))))*(x(N))));
    q(N-2) = (-0.0833333333333333333333333333333)*((pow(h,-1))*((((-1)*(x(0)))+((x((-4)+(N)))+(((-8)*(x((-3)+(N))))+((8)*(x((-1)+(N)))))))*(x((-2)+((2)*(N))))));
    q(N-1) = (-0.0833333333333333333333333333333)*((pow(h,-1))*((((8)*(x(0)))+(((-1)*(x(1)))+((x((-3)+(N)))+((-8)*(x((-2)+(N)))))))*(x((-1)+((2)*(N))))));

    return q;
  }

  
  // static Vector compute_energy_density(Workspace &workspace, const double t);
  
  // static Vector compute_momentum_density(Workspace &workspace, const double t);
};

struct SineGordon1DBooster {
  typedef Eigen::ArrayXd State;
  typedef Eigen::ArrayXd Vector;
  typedef SineGordonParam Param;
  // typedef WorkspaceGeneric<Vector, State> Workspace;
  // Workspace &workspace;
  Param param;
  Vector tau;
  
  // SineGordon1DEquation(Workspace &workspace_) : workspace(workspace_) {}
  SineGordon1DBooster(Param param_, const Vector &tau_) : param(param_), tau(tau_) {}

  // TODO
  void operator()(const State &x, State &dxdt, const double) {
    using namespace Eigen;
    const long long int N = param.N;
    const double L = param.L;
    const double h = L / N;
    auto u = x(seqN(0, N));
    auto dudt = x(seqN(N, N));

    dxdt(seqN(0, N)) = tau * dudt;
    dxdt(seqN(N+2, N-4)) = ((((-1)*(sin(x(seqN((2),((-4)+(N)))))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((-1)*(x(seqN((0),((-4)+(N))))))+(((16)*(x(seqN((1),((-4)+(N))))))+(((-30)*(x(seqN((2),((-4)+(N))))))+(((16)*(x(seqN((3),((-4)+(N))))))+((-1)*(x(seqN((4),((-4)+(N)))))))))))))*(tau(seqN((2),((-4)+(N))))))+((0.00694444444444444444444444444444)*((pow(h,-2))*(((x(seqN((0),((-4)+(N)))))+(((-8)*(x(seqN((1),((-4)+(N))))))+(((8)*(x(seqN((3),((-4)+(N))))))+((-1)*(x(seqN((4),((-4)+(N)))))))))*((tau(seqN((0),((-4)+(N)))))+(((-8)*(tau(seqN((1),((-4)+(N))))))+(((8)*(tau(seqN((3),((-4)+(N))))))+((-1)*(tau(seqN((4),((-4)+(N))))))))))));
    dxdt(N+1) = ((((-1)*(sin(x(1))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((16)*(x(0)))+(((-30)*(x(1)))+(((16)*(x(2)))+(((-1)*(x(3)))+((-1)*(x((-1)+(N)))))))))))*(tau(1)))+((0.00694444444444444444444444444444)*((pow(h,-2))*((((-8)*(x(0)))+(((8)*(x(2)))+(((-1)*(x(3)))+(x((-1)+(N))))))*(((-8)*(tau(0)))+(((8)*(tau(2)))+(((-1)*(tau(3)))+(tau((-1)+(N)))))))));
    dxdt(N+0) = ((((-1)*(sin(x(0))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((-30)*(x(0)))+(((16)*(x(1)))+(((-1)*(x(2)))+(((-1)*(x((-2)+(N))))+((16)*(x((-1)+(N)))))))))))*(tau(0)))+((0.00694444444444444444444444444444)*((pow(h,-2))*((((8)*(x(1)))+(((-1)*(x(2)))+((x((-2)+(N)))+((-8)*(x((-1)+(N)))))))*(((8)*(tau(1)))+(((-1)*(tau(2)))+((tau((-2)+(N)))+((-8)*(tau((-1)+(N))))))))));
    dxdt(N+N-2) = ((((-1)*(sin(x((-2)+(N)))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((-1)*(x(0)))+(((-1)*(x((-4)+(N))))+(((16)*(x((-3)+(N))))+(((-30)*(x((-2)+(N))))+((16)*(x((-1)+(N)))))))))))*(tau((-2)+(N))))+((0.00694444444444444444444444444444)*((pow(h,-2))*((((-1)*(x(0)))+((x((-4)+(N)))+(((-8)*(x((-3)+(N))))+((8)*(x((-1)+(N)))))))*(((-1)*(tau(0)))+((tau((-4)+(N)))+(((-8)*(tau((-3)+(N))))+((8)*(tau((-1)+(N))))))))));
    dxdt(N+N-1) = ((0.00694444444444444444444444444444)*((pow(h,-2))*((((8)*(x(0)))+(((-1)*(x(1)))+((x((-3)+(N)))+((-8)*(x((-2)+(N)))))))*(((8)*(tau(0)))+(((-1)*(tau(1)))+((tau((-3)+(N)))+((-8)*(tau((-2)+(N))))))))))+((((-1)*(sin(x((-1)+(N)))))+((0.0833333333333333333333333333333)*((pow(h,-2))*(((16)*(x(0)))+(((-1)*(x(1)))+(((-1)*(x((-3)+(N))))+(((16)*(x((-2)+(N))))+((-30)*(x((-1)+(N)))))))))))*(tau((-1)+(N))));
  }
  
};

#endif
