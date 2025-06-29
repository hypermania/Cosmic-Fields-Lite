/*!
  \file sine_gordon_1d.hpp
  \author Siyang Ling
  \brief Tools related to 1D Sine-Gordon fields.

  This is a standalone header module for 1D Sine-Gordon fields.
  Implemented functionalities include the 1D Sine-Gordon equation (on a periodic interval), functions to compute the stress-energy of the field (see SineGordon1DEquation), and the spatially varying boost on the field (see SineGordon1DBooster).
*/
#ifndef SINE_GORDON_1D_HPP
#define SINE_GORDON_1D_HPP

#include "Eigen/Dense"

struct SineGordonParam {
  long long int N; /*!< Number of lattice points of the periodic interval. */
  double L; /*!< Length of the periodic interval. */
  double v;
};

/*! 
  \brief The 1D Sine-Gordon system, \f$ \partial_t^2 u - \partial_x^2 u + \sin(u) = 0 \f$.
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

  /*!
    \brief The function called by odeint library.
    \param[in] x The current state of the Sine-Gordon system, in the form of a length \f$2N\f$ ArrayXd. Convention: [u, dudt].
    \param[out] dxdt The time derivative, dxdt of the system.
    \param t The current time parameter.
  */
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

  /*!
    \brief Given the current field state, returns the pointwise energy density of a Sine-Gordon system.
  */
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
  
  /*!
    \brief Given the current field state, returns the pointwise momentum density of a Sine-Gordon system.
  */
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
  
  /*!
    \brief Given the current field state, returns the pointwise pressure density of a Sine-Gordon system.
  */
  Vector compute_pressure(const State &x, const double t) {
    using namespace Eigen;
    const long long int N = param.N;
    const double L = param.L;
    const double h = L / N;
    Vector p(N);


    p(seqN(2, N-4)) = (-1)+((cos(x(seqN((2),((-4)+(N))))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow((x(seqN((0),((-4)+(N)))))+(((-8)*(x(seqN((1),((-4)+(N))))))+(((8)*(x(seqN((3),((-4)+(N))))))+((-1)*(x(seqN((4),((-4)+(N)))))))),2))))+((0.500000000000000000000000000000)*(pow(x(seqN(((2)+(N)),((-4)+(N)))),2)))));
    p(1) = (-1)+((cos(x(1)))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((-8)*(x(0)))+(((8)*(x(2)))+(((-1)*(x(3)))+(x((-1)+(N))))),2))))+((0.500000000000000000000000000000)*(pow(x((1)+(N)),2)))));
    p(0) = (-1)+((cos(x(0)))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((8)*(x(1)))+(((-1)*(x(2)))+((x((-2)+(N)))+((-8)*(x((-1)+(N)))))),2))))+((0.500000000000000000000000000000)*(pow(x(N),2)))));
    p(N-2) = (-1)+((cos(x((-2)+(N))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((-1)*(x(0)))+((x((-4)+(N)))+(((-8)*(x((-3)+(N))))+((8)*(x((-1)+(N)))))),2))))+((0.500000000000000000000000000000)*(pow(x((-2)+((2)*(N))),2)))));
    p(N-1) = (-1)+((cos(x((-1)+(N))))+(((0.00347222222222222222222222222222)*((pow(h,-2))*(pow(((8)*(x(0)))+(((-1)*(x(1)))+((x((-3)+(N)))+((-8)*(x((-2)+(N)))))),2))))+((0.500000000000000000000000000000)*(pow(x((-1)+((2)*(N))),2)))));

    return p;
  }

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
