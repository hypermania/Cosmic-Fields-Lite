# Progress Log

## 2026-04-27

- Problem encountered: The documented CPU build (`make -j2 disable-cuda=true`) could not complete in this environment because `fftw3.h` is not installed under the configured include paths.
- How it was solved: The full build was left blocked as an environment dependency issue, and a focused `make test` target was added for the I/O code path that does not require FFTW.
- How to avoid it in future: Install FFTW3 development headers/libraries or pass correct `FFTW_INCLUDE_DIR` and `FFTW_LIBRARY_DIR` values before running the full build.
- Problem encountered: Binary I/O helpers silently accepted missing/truncated files, `load_vector_from_file` leaked its raw read buffer, and `write_to_filename_template` used a fixed 128-byte `sprintf` buffer.
- How it was solved: Commit `f7a7699a` replaced raw-buffer reads with direct vector reads, added file open/read/write validation, dynamically sizes formatted filenames, declares `load_vector_from_file`, adds `make test`, and documents the codebase in `TECHNICAL_REPORT.md`.
- How to avoid it in future: Keep binary I/O code covered by tests for round trips, missing files, malformed sizes, and long generated paths.
- Git commit ID: `f7a7699a`
- Problem encountered: After FFTW 3.3.10 was installed, the technical report still described the CPU build as blocked by missing `fftw3.h`; additionally, `make -j2 disable-cuda=true` exceeded this machine's compile-time memory budget while building `src/field_booster.cpp`.
- How it was solved: Commit `3ab17f82` updated the report to record the installed FFTW path, passing `make test`, the `make -j2` memory limitation, and the successful `make -j1 disable-cuda=true` CPU link.
- How to avoid it in future: Re-run verification after dependency changes and document the exact build parallelism that is known to pass on the current machine.
- Git commit ID: `3ab17f82`
