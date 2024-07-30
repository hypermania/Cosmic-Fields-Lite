# Writing your own equation

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
When this function is called, it computes the time derivative of the state vector \f$ x \f$ at time \f$ t \f$, and stores it to \f$ dxdt \f$.
Implementing this function is the minimal requirement for a class to work with odeint.
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
The code given above won't compile since `workspace.kappa` doesn't exist yet.


## Adding a new parameter
You will need to define a new parameter struct that contains the new \f$ \kappa \f$ parameter.


## Add a function to compute energy density
In order to save density spectrum, you would also want to implement a function to calculate energy density profile.


## Using CUDA
You want your equation to work on GPU memory, so the state vector would be:
```{.cpp}
typedef thrust::device_vector<double> Vector;
```
Your `operator()` should modify the state vector.
To do that, the easiest way is to write your own CUDA kernel.
