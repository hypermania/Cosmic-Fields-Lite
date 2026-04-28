# Technical Report: Cosmic-Fields-Lite

## Repository Snapshot

Cosmic-Fields-Lite is a C++20 framework for lattice field simulations in cosmology. The codebase targets both CPU and CUDA execution, with Eigen vectors used for CPU state storage and `thrust::device_vector<double>` used for GPU state storage. The default branch inspected was `release` from `hypermania/Cosmic-Fields-Lite`, cloned through the `hypermania-bot` fork.

The repository is source-heavy and compact: the simulation framework is about 8.3k lines under `src/`, with vendored Eigen and Boost headers under `external/`. The project also includes a `documentation.pdf`, Doxygen configuration, Mathematica notebooks, a plotting utility, and tracked sample output under `output/Growth_and_FS/`.

## Build And Runtime Dependencies

The project uses a root `Makefile`.

Required:

- C++20 compiler.
- FFTW3 headers and library.
- Eigen 3.4 and Boost 1.84, both vendored in `external/`.

Optional:

- CUDA Toolkit for GPU equations, CUFFT wrappers, and Thrust-backed odeint integration.

The documented CPU-only build command is:

```sh
make -j disable-cuda=true
```

In this environment, FFTW 3.3.10 is installed under `/usr/local`, which matches the Makefile defaults for `FFTW_INCLUDE_DIR` and `FFTW_LIBRARY_DIR`. The CPU executable links successfully with `make -j1 disable-cuda=true`.

A focused I/O test target was added and does not require FFTW:

```sh
make test
```

## High-Level Architecture

The central design is a small set of generic infrastructure types plus equation-specific implementations.

- `WorkspaceGeneric` in `src/workspace.hpp` owns simulation state, lattice parameters, temporary fields, FFT wrappers, and optional approximation data. It is templated over vector and state types, which is how the same simulation structure can support Eigen CPU vectors and Thrust GPU vectors.
- `initializer.hpp` defines inline initializer lambdas. They create initial conditions on the CPU, then copy into the workspace vector type through `ALGORITHM_NAMESPACE`, which resolves to `std` or `thrust`.
- `equations.hpp` and `equations.cpp` define CPU equations for Klein-Gordon, Klein-Gordon in FRW, and comoving-curvature-coupled scalar evolution.
- CUDA variants live in `equations_cuda.cu/.cuh`, with supporting wrappers in `cuda_wrapper.cu/.cuh` and `fdm3d_cuda.cu/.cuh`.
- `observer.hpp` implements odeint observers that periodically write field spectra, density spectra, density slices, and time lists.
- `fdm3d.cpp/.hpp` contains lattice indexing, spectral summaries, inverse Laplacian, Fourier cutoff, and Fourier-mode scaling helpers.
- `fftw_wrapper.cpp/.hpp` wraps FFTW plans for repeated real-to-complex and complex-to-real transforms.
- `field_booster.cpp/.hpp`, `proca.*`, `sp.*`, and `sine_gordon_1d.hpp` implement newer spatially varying boost examples and additional field systems.

## Execution Flow

`src/main.cpp` acts as a scenario driver rather than a command-line application. Most available workflows are hard-coded functions that users uncomment or edit:

- `solve_field_equation()` runs the field simulation associated with the wave dark matter free-streaming study.
- `generate_wkb_solutions()` extends a saved state with WKB evolution.
- `generate_ic_kg()`, `generate_ic_proca()`, `generate_ic_sp()`, and `generate_ic_sg()` generate spatially varying boost initial conditions.

The currently active main path calls `generate_ic_sg()`.

A typical simulation flow is:

1. Construct a parameter POD such as `MyParam`.
2. Construct a `WorkspaceGeneric` with an initializer lambda.
3. Construct an equation object bound to the workspace.
4. Construct a `ConstIntervalObserver`.
5. Run Boost odeint over `workspace.state`.
6. Write spectra, slices, parameters, and final state to `output/`.

## Data Model

Most scalar field states are stored as concatenated vectors:

- First `N^3` entries: field value.
- Second `N^3` entries: field time derivative.

Some systems extend this pattern. Proca fields use multiple vector components, and Schroedinger-Poisson uses complex array state in the newer generic workspace path.

Lattice indexing uses row-major macros in `fdm3d.hpp`:

- `IDX_OF(N, i, j, k)` for dense real-space fields.
- `PADDED_IDX_OF(N, i, j, k)` for FFTW/CUFFT padded real-output layouts.

Binary output is raw little-endian native double data, with separate parameter name/type metadata intended for Mathematica or Python readers.

## Strengths

- The CPU/GPU split is relatively clean. Dispatch is concentrated in `dispatcher.hpp`, wrapper classes, and initializer copy paths.
- The workspace pattern keeps large temporary buffers reusable and avoids passing many simulation objects through every call.
- Equation implementations are explicit and easy to map to the physics formulas.
- FFT planning is amortized through wrapper objects rather than recreated for every transform.
- The codebase includes domain-specific examples for current research workflows instead of only abstract framework pieces.

## Risks And Maintainability Issues

- The root `main.cpp` is an editable experiment driver. This is practical for research, but it makes reproducible scenario selection harder than a small CLI or config layer.
- Build configuration is manually path-driven. FFTW defaults point to `/usr/local`, which is brittle on Linux distributions where FFTW is commonly under `/usr/include` and `/usr/lib`.
- Several APIs silently assumed successful file I/O before the bug fix described below.
- Many lattice loops use `int` in newer modules even though project parameters are `long long int`. Current documented sizes are safe, but this weakens the generic contract for larger grids.
- Output files are native binary doubles without endianness/version metadata. That is fast, but fragile for long-term archival.
- There is little automated test coverage. The new I/O test is a starting point, not a numerical regression suite.

## Bug Fixed During Review

The binary I/O helpers had three related defects:

- `load_vector_from_file` allocated a raw character buffer and never freed it.
- Both loaders silently accepted missing files and attempted to proceed with zero-length data.
- `write_to_filename_template` formatted paths into a fixed 128-byte stack buffer with `sprintf`, which could overflow for long output directories.

The fix:

- Replaced manual heap buffering with direct reads into `std::vector<double>` and `Eigen::VectorXd`.
- Added read/write open checks, read/write failure checks, and validation that binary file sizes are multiples of `sizeof(double)`.
- Declared `load_vector_from_file` in `io.hpp`.
- Replaced fixed-buffer `sprintf` with dynamically sized `std::snprintf`.
- Added `make test`, backed by `tests/io_test.cpp`, covering round-trip loads, long templated filenames, and missing-file error handling.

## Verification

Commands run:

```sh
make test
```

Result: passed.

```sh
make -j2 disable-cuda=true
```

Result: the parallel build was killed while compiling `src/field_booster.cpp`, consistent with memory pressure on this machine rather than a source or dependency error.

```sh
make -j1 disable-cuda=true
```

Result: passed and linked `main` against `/usr/local/lib/libfftw3`.
