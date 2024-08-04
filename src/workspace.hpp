/*!
  \file workspace.hpp
  \author Siyang Ling
  \brief A generic "workspace" class, containing parameters / data / tools used during simulations.
*/
#ifndef WORKSPACE_HPP
#define WORKSPACE_HPP

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


/*!
  \brief A generic workspace for storing temporary objects within simulation.

  WorkspaceGeneric contains everything used during simulations, including the field state, gravitational potential, parameters, etc.
  The numerical integrator, observers or other utilities will read from or write to one `WorkspaceGeneric` instance, 
  so that data can be shared between different functionalities.
  The user is responsible for maintaining the members variables within the workspace.
  
  At construction, `WorkspaceGeneric` takes in a `param` struct (containing just a few numbers) and an `initializer` function (see initializer.hpp).
  Typically the values in `param` are copied to fields in the workspace with the same name.
  The initializer then use the `param` and its own logic to fill in the workspace. (e.g. initial conditions, curvature perturbations)
  Note that the `Vector`'s in the workspace are initially empty, and they need to be resized (via `vec.resize()`) to be written to.
*/
template<typename Vector>
struct WorkspaceGeneric {
  typedef Vector State;
  long long int N; /*!< Number of lattice points. */
  double L; /*!< Box size. */
  double m; /*!< Mass of field. */
  StaticEOSCosmology cosmology{}; /*!< FRW cosmology. */
  State state; /*!< The full equation state, usually a vector like \f$ (\varphi, \dot{\varphi}) \f$ (for 2nd order equations). */
  double lambda{0}; /*!< Quartic self-interaction coupling constant. */
  double f_a{1.0}; /*!< A scale in the monodromy potential. */
  Vector Psi;
  Vector dPsidt;
  Vector Psi_fft;
  Vector dPsidt_fft;
  Vector R_fft; /*!< Usually used to store comoving curvature perturbations. */
  std::vector<double> t_list; /*!< The list of coordinate times at which a save is stored. */
  typename fftWrapperDispatcher<Vector>::Generic fft_wrapper; /*!< A FFT wrapper for 3D lattice with size \f$ N \f$. */

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
