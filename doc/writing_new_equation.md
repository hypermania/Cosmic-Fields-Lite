# Implementing your own equation

Here we give an example of adding a field equation with \f$ \kappa \varphi^6 \f$ interaction to the codebase.
\f[ \ddot{\varphi} - \nabla^2 \varphi + m^2 \varphi + \kappa \varphi^5 = 0 \f]


## Adding the equation class
We use the boost odeint library for numerical integration. To use the library, we need to implement a new equation class.
See [this link](https://www.boost.org/doc/libs/1_85_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/getting_started/short_example.html) for an example of odeint equation class.
In our case, the equation class with \f$ \varphi^6 \f$ looks like:
```{.cpp}
struct KappaEquation {
  typedef Eigen::VectorXd Vector;
  typedef Vector State;
  typedef WorkspaceGeneric<State> Workspace;
  Workspace &workspace;
  
  KappaEquation(Workspace &workspace_) : workspace(workspace_) {}

  void operator()(const State &dxdt, State &x, const double t);
};
```
In the first few lines, the types `Vector`, `State` and `Workspace` are defined.
These definitions specify what state vector the equation is going to work with: if you want to use different state vector types (e.g. GPU device vector), you will need to define different equation classes.
Here we use `Eigen::VectorXd`.
The equation class also has a reference to a `workspace`, so that it has access to essential information for evolution (e.g. mass and coupling parameters).


The most important function here is the `operator()`.
When this function is called, it computes the time derivative of the state vector `x` at time `t`, and stores it to `dxdt`.
Implementing this function is the minimal requirement for a class to work with the odeint library.
To do this, we can simply copy the implementation for `KleinGordonEquation::operator()` and add a \f$ \kappa \varphi^5 \f$ term to it:
```{.cpp}
void KappaEquation::operator()(const State &x, State &dxdt, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double kappa = workspace.kappa;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      dxdt(seqN(IDX_OF(N, a, b, 0), N)) = x(seqN(N*N*N+IDX_OF(N, a, b, 0), N));
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 0), N)) =
	(-1.0) * m * m * x(seqN(IDX_OF(N, a, b, 0), N))
	- kappa * pow(x(seqN(IDX_OF(N, a, b, 0), N)), 5)
	+ inv_h_sqr * (-6.0 * x(seqN(IDX_OF(N, a, b, 0), N))
		       + x(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
		       + x(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))
		       + x(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
		       + x(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N)));
      dxdt(seqN(N*N*N+IDX_OF(N, a, b, 1), N-2)) +=
	inv_h_sqr * ( x(seqN(IDX_OF(N, a, b, 2), N-2))
		      + x(seqN(IDX_OF(N, a, b, 0), N-2)) );
      dxdt(N*N*N+IDX_OF(N, a, b, 0)) +=
	inv_h_sqr * ( x(IDX_OF(N, a, b, N-1)) + x(IDX_OF(N, a, b, 1)) );
      dxdt(N*N*N+IDX_OF(N, a, b, N-1)) +=
	inv_h_sqr * ( x(IDX_OF(N, a, b, N-2)) + x(IDX_OF(N, a, b, 0)) );
    }
  }
}
```
Note the extra line `kappa * pow(x(seqN(IDX_OF(N, a, b, 0), N)), 5)` giving the \f$ \kappa \varphi^5 \f$ term in the equation.


## Adding the coupling parameter in workspace
The code given above won't compile yet since `workspace.kappa` doesn't exist.
To make the code compile, add a new field in `WorkspaceGeneric`:
```{.cpp}
template<typename Vector>
struct WorkspaceGeneric {
	// Stuff
	double kappa;
	// Stuff
};
```
As a general paradigm, we put data (e.g. coupling parameters, temporary variables) needed to solve the equation in a `Workspace`.
Note that different equations use the same `Workspace`, and the field names (e.g. `kappa`) can mean different things for different equations.
You are responsible for ensuring that your modification on `Workspace` doesn't introduce bugs for other equations.
To avoid accidently introducing a bug, you are advised to add new fields for new parameters / temporary objects.


## Setting `workspace.kappa` from a parameter struct
Now suppose you define a new parameter class:
```{.cpp}
struct KappaParam {
	// The usual
	double kappa;
};

KappaParam param;
```
If you try calling the constructor `Workspace(param, initializer)`, `workspace.kappa` would not be automatically set to `param.kappa`.
To resolve this, add the following in `workspace.hpp`:
```{.cpp}
template<typename Param>
concept HasKappa = requires (Param param) { TYPE_REQUIREMENT(param.kappa, double) };

// ...

    WorkspaceGeneric(const Param &param, auto &initializer) :
    N(param.N), L(param.L), fft_wrapper(param.N)
  {
    if constexpr(HasKappa<Param>) { kappa = param.kappa; }
	// ...
  }
```
This piece of code uses concept `HasKappa` to detect if `param.kappa` exists or not, and set `workspace.kappa` to `param.kappa` in the case it exists.
Having `KappaParam` is useful since it works with the utilities in `param.h`.


## Add a function to compute energy density
In order to save density spectrum, you would also want to implement a function to calculate energy density profile.
Again we can imitate the implementation for `KleinGordonEquation`:
```{.cpp}
struct KappaEquation {
	// ...
	static Vector compute_energy_density(const Workspace &workspace, const double t);
};

KappaEquation::Vector KappaEquation::compute_energy_density(const Workspace &workspace, const double t)
{
  using namespace Eigen;
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double kappa = workspace.kappa;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));
  
  VectorXd rho(workspace.state.size() / 2);
    
  for(long long int a = 0; a < N; ++a){
    for(long long int b = 0; b < N; ++b){
      rho(seqN(IDX_OF(N, a, b, 0), N)) = 0.5 *
	( workspace.state(seqN(N*N*N+IDX_OF(N, a, b, 0), N)).cwiseAbs2()
	  + m * m * workspace.state(seqN(IDX_OF(N, a, b, 0), N)).cwiseAbs2()
	  + (1.0 / 6.0) * kappa * pow(workspace.state(seqN(IDX_OF(N, a, b, 0), N)), 6)
	  + 0.25 * inv_h_sqr *
	  ( (workspace.state(seqN(IDX_OF(N, (a+1)%N, b, 0), N))
	     - workspace.state(seqN(IDX_OF(N, (a+N-1)%N, b, 0), N))).cwiseAbs2()
	    + (workspace.state(seqN(IDX_OF(N, a, (b+1)%N, 0), N))
	       - workspace.state(seqN(IDX_OF(N, a, (b+N-1)%N, 0), N))).cwiseAbs2() )
	  );
      rho(seqN(IDX_OF(N, a, b, 1), N-2)) += 0.5 * 0.25 * inv_h_sqr *
	(workspace.state(seqN(IDX_OF(N, a, b, 2), N-2))
	 - workspace.state(seqN(IDX_OF(N, a, b, 0), N-2))).cwiseAbs2();
      rho(IDX_OF(N, a, b, 0)) += 0.5 * 0.25 * inv_h_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 1)) - workspace.state(IDX_OF(N, a, b, N-1)), 2);
      rho(IDX_OF(N, a, b, N-1)) += 0.5 * 0.25 * inv_h_sqr *
	pow(workspace.state(IDX_OF(N, a, b, 0)) - workspace.state(IDX_OF(N, a, b, N-2)), 2);
    }
  }
  return rho;
}
```
Note the extra \f$ \kappa \varphi^6 / 6\f$ term in the function above.
Now `ConstIntervalObserver` knows how to compute the energy density for this theory.

## Using CUDA
If you want your equation to run on GPU memory, then in `KappaEquation` the vector type should be set to:
```{.cpp}
typedef thrust::device_vector<double> Vector;
```
Here, `thrust::device_vector<double>` is a class in the `thrust` library representing a double floating point array on the GPU.
Much like `std::vector<double>`, the class `thrust::device_vector<double>` takes care of GPU memory allocation / deallocation in an RAII manner, so that you don't have to call CUDA memory management API directly.
See [thrust documentation](https://nvidia.github.io/cccl/thrust/api/classthrust_1_1device__vector.html) for more details.

Your `operator()` and `compute_energy_density` functions must now work on GPU device vectors.
A straightforward way to do this is to write your own CUDA kernel.
Here's an example on how to do it:
```{.cpp}
__global__
void kappa_equation_kernel(const double *x, double *dxdt,
	const double m, const double kappa,
	const double inv_h_sqr,
	const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  
  dxdt[IDX_OF(N, a, b, c)] = x[N*N*N+IDX_OF(N, a, b, c)];
  dxdt[N*N*N+IDX_OF(N, a, b, c)] =
    - m * m * x[IDX_OF(N, a, b, c)]
	- kappa * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)]
    + inv_h_sqr * (-6.0 * x[IDX_OF(N, a, b, c)]
		    + x[IDX_OF(N, (a+1)%N, b, c)]
		    + x[IDX_OF(N, (a+N-1)%N, b, c)]
		    + x[IDX_OF(N, a, (b+1)%N, c)]
		    + x[IDX_OF(N, a, (b+N-1)%N, c)]
		    + x[IDX_OF(N, a, b, (c+1)%N)]
		    + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}

void KappaEquation::operator()(const State &x, State &dxdt, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double kappa = workspace.kappa;
  const double inv_h_sqr = 1.0 / ((L / N) * (L / N));

  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  kappa_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, kappa, inv_h_sqr, N);
}
```
Here `kappa_equation_kernel` is the CUDA kernel, and the `__global__` specifier means this function runs on the GPU.
`KappaEquation::operator()` invokes the kernel via `kappa_equation_kernel<<<numBlocks, threadsPerBlock>>>`. 
Given the execution configuration `threadsPerBlock` and `numBlocks`, the function `kappa_equation_kernel` is executed once for each `a,b,c`, with `0 <= a,b,c < N`.
Depending on the kernel, different execution configurations could result in varying performance, or even introduce bugs.
See the [CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) for more details.
