# Writing your own equation

Here we give an example of adding a field equation with \f$ \kappa \phi^6 \f$ interaction to the codebase.

## Adding a new parameter
You will need to define a new parameter struct that contains the new \f$ \kappa \f$ parameter.

## Adding the equation class
```{.cpp}
struct KappaEquation;
```

### The equation function
You need to define at least one function in the 
The `odeint` library

### The energy density function
