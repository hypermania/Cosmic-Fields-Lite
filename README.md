# lite-cosmic-sim

**lite-cosmic-sim** is a lightweight and modular framework for performing field simulations in cosmology. This framework was used for studying free-streaming of wave dark matter; see [arXiv:XXXX.XXXX](https://arxiv.org) for the study. The codebase contains several field equations on both CPU and GPU (CUDA), offering choices for numerical methods and simulation outputs.

## Overview
This codebase aims to be:

1. As fast as possible. Users should be able to write code that exhausts hardward potential within this framework.
2. Easily modifiable and extensible. Users should be able to focus on physics-relevant code, such as that for setting initial conditions or the field equation.

To achieve these goals, the framework is written in a modular structure. This allows users to easily switch between different initial conditions, field equations, output methods, and even between using CPUs or GPUs for computation. Users have to and only have to provide the low level implementation for the physics-relevant code. This means users have full control over optimization of core routines, and they are not limited to a specific set of provided features. This flexibility makes it easy for the user to test new ideas, which is useful in research.

## Sample usage
The following code initializes a homogeneous Klein Gordon field with (initially) unit field strength and zero time derivative. Then the field is evolved from `t=0` to `t=10`. Field and density spectra are saved to disk per unit time.
```C++
#include "param.hpp"
#include "initializer.hpp"
#include "equations.hpp"
#include "observer.hpp"

struct MyParam {
  long long int N = 256; // Number of lattice sites (per axis)
  double L = 256.0; // Box size
  double m = 1.0; // Field mass
  double f = 1.0; // The initial (homogeneous) field value
  double dt_f = 0.0; // The initial (homogeneous) field time derivative value
  double t_start = 0.0; // Start time of numerical integration
  double t_end = 10.0; // End time of numerical integration
  double t_interval = 1.0; // Interval between saving outputs
};

int main() {
  using namespace Eigen;
  using namespace boost::numeric::odeint;

  typedef KleinGordonEquation Equation;
  typedef Eigen::VectorXd State;
  typedef WorkspaceGeneric<State> Workspace;
  
  MyParam param;
  
  Workspace workspace(param, homogeneous_field);
  
  Equation eqn(workspace);

  ConstIntervalObserver<Equation> observer("output/sample_equation/", param, eqn);

  auto stepper = runge_kutta4_classic<State>();

  integrate_const(stepper, eqn, workspace.state, param.t_start, param.t_end, 0.1, observer);
}
```

Here's a break down of the code:

* `MyParam` is a POD struct specifying parameters for the simulation. You may define your own struct to include new parameters (coupling strength, FRW universe parameters, time step size, etc), as long as it is a POD and contains lattice parameters `N` and `L`.
* `Workspace` is a type containing temporary variables for a simulation (e.g. the field). It is initialized with `param` and a callback `homogeneous_field`, which sets the field to homogeneous value `param.f` and time derivative `param.dt_f`. You can easily define your own callbacks (using lambdas) to set other sorts of initial conditions.
* `Equation` is the equation to be solved. Here it is the pre-defined `KleinGordonEquation`. You can of course write your own equations.
* `ConstIntervalObserver<Equation>` specifies how to save outputs during simulation. By default it saves spectra for field and density.
* `stepper` is the RK4 method provided by the [boost odeint](https://www.boost.org/doc/libs/1_85_0/libs/numeric/odeint/doc/html/index.html) library. You can choose other methods (e.g. Euler, DOPRI5) in the library, or even write your own. The `odeint` library is responsible for the main numerical integration loop in this codebase.
* `integrate_const` is a convenience function in the `odeint` library. It runs the simulation and saves results to "output/sample_equation", as specified by `observer`.



## How to build the project
Compiler requirement: a C++ compiler supporting C++20. (I used g++ 12.2.0.)

Required dependency: [fftw3](https://www.fftw.org/fftw3_doc/index.html)

Optional dependency: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

We also included these header-only libraries [Eigen 3.4.0](https://eigen.tuxfamily.org) and [boost 1.84](https://www.boost.org/) along with the codebase in the `external` directory.

`Makefile` is used for build system. We have tested compilation on Linux and MacOS systems. To compile the project:

* (If default settings don't work:) Modify the `Makefile` so that it knows where your fftw or CUDA include files / library files are.
* If you have CUDA Toolkit installed, simply run `make`.
* If you don't have CUDA Toolkit, run `make disable-cuda=true`. (We use compiler flags to comment out CUDA-dependent code.  e.g. CudaComovingCurvatureEquationInFRW)

**Note: If you have a CUDA compatible NVIDIA GPU, using CUDA is highly recommended. In our case, it produced more than 10 times speedup.**

## Documentation
If you have doxygen, you can build the documentation by running `doxygen doxygen.config`.

## Utilities for interpreting output
We include two Mathematica notebooks `numeric_plot.nb`, `animation.nb` and a python utility `plot_util.py` for plotting results.

## Overview of implemented functionalities
| Symbol                                                                                                                                                   |                                                                                                                                                                               Description                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  `generate_inhomogeneous_gaussian_random_field`                                                                                                          |Functionality for initializing Gaussian random fields with spatially inhomogeneous variances. This procedure is crucial for generating the initial conditions used in the paper.                                                           |
|`KleinGordonEquationInFRW` and `CudaKleinGordonEquationInFRW`                                                                                             |Klein Gordon equation that runs on CPU and GPU. Used in section 4.2.1 of paper.                                                                                                                                                            |
|`ComovingCurvatureEquationInFRW`, `CudaComovingCurvatureEquationInFRW` and `CudaApproximateComovingCurvatureEquationInFRW`                                |A scalar field in the presence of external gravity that is consistent with some set of comoving curvature perturbations. Used in section 4.2.2 of paper.                                                                                   |
|`CudaSqrtPotentialEquationInFRW`                                                                                                                          |A scalar field with monodromy potential. Used in section 4.2.3 of paper.                                                                                                                                                                   |
|                                                                                                                         `CudaFixedCurvatureEquationInFRW`|                                                                                                                                                                                         A scalar field in a fixed gravitational potential.|
|                                                                                                                             `CudaLambdaEquationInFRW`    | A scalar field with lambda phi^4 interaction.                                                                                                                                                                                             |



## Notes on using CUDA
We use the `thrust` library (included in the CUDA Toolkit) for allocating/deallocating GPU memory. We also separate compilation of `.cpp` files and `.cu` files, so that users without CUDA can also use the code.