/*!
  \file workspace.hpp
  \brief A generic "workspace" class, containing parameters / data / tools used during simulations.
  
  WorkspaceGeneric contains everything used during simulations, including the field state, gravitational potential, parameters, etc.
  It is initialized by a Param struct (containing just a few numbers) and an "initializer" (see initializer.hpp).
*/
#ifndef WORKSPACE_H
#define WORKSPACE_H

#include <memory>

#include "param.hpp"
#include "physics.hpp"
#include "fftw_wrapper.hpp"
#include "dispatcher.hpp"

#define TYPE_REQUIREMENT(value, type) {std::remove_cvref_t<decltype((value))>()} -> std::same_as<type>;



template<typename Param>
concept HasLatticeParams = requires (Param param)
  { TYPE_REQUIREMENT(param.N, long long int)
    TYPE_REQUIREMENT(param.L, double) };

template<typename Param>
concept HasMass = requires (Param param) { TYPE_REQUIREMENT(param.m, double) };

template<typename Param>
concept HasLambda = requires (Param param) { TYPE_REQUIREMENT(param.lambda, double) };

template<typename Param>
concept HasFa = requires (Param param) { TYPE_REQUIREMENT(param.f_a, double) };

template<typename Param>
concept HasFRWParameters = requires (Param param)
  { TYPE_REQUIREMENT(param.a1, double)
    TYPE_REQUIREMENT(param.H1, double)
    TYPE_REQUIREMENT(param.t1, double) };

template<typename Param>
concept HasPsiApproximationParameters = requires (Param param)
  { TYPE_REQUIREMENT(param.M, long long int) };


/* 
   The workspace for solving equations.
   The lifetime of objects in the workspace are managed by us (this codebase), instead of external libraries (e.g. odeint).
*/
template<typename Vector>
struct WorkspaceGeneric {
  typedef Vector State;
  long long int N;
  double L;
  double m;
  StaticEOSCosmology cosmology{};
  State state;
  double lambda{0};
  double f_a{1.0};
  Vector Psi;
  Vector dPsidt;
  Vector Psi_fft;
  Vector dPsidt_fft;
  Vector R_fft;
  std::vector<double> t_list;
  typename fftWrapperDispatcher<Vector>::Generic fft_wrapper;

  bool Psi_approximation_initialized{false};
  long long int M;
  std::unique_ptr<typename fftWrapperDispatcher<Vector>::Generic> fft_wrapper_M_ptr;
  Vector cutoff_R_fft;
  
  template<HasLatticeParams Param>
  WorkspaceGeneric(const Param &param, auto &initializer) :
    N(param.N), L(param.L), fft_wrapper(param.N)
  {
    //static_assert(HasLatticeParams<Param>, "HasLatticeParams<Param> test failed.");
    if constexpr(HasFRWParameters<Param>) { cosmology = StaticEOSCosmology(param); }
    if constexpr(HasMass<Param>) { m = param.m; }
    if constexpr(HasLambda<Param>) { lambda = param.lambda; }
    if constexpr(HasFa<Param>) { f_a = param.f_a; }
    if constexpr(HasPsiApproximationParameters<Param>) { M = param.M;
	assert(N >= M); }
    initializer(param, *this);
  }
};




#endif
