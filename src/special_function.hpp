#ifndef SPECIAL_FUNCTION_HPP
#define SPECIAL_FUNCTION_HPP

// Pade approximant for Si(x), with m=15, n=12
inline double Si_pade_approximant_15_12(double x) {
  using namespace std;
  return (x - 0.045439340981633 * pow(x, 3) + 0.0011545722575101668 * pow(x, 5) -
          0.000014101853682133025 * pow(x, 7) + 9.432808094387131e-8 * pow(x, 9) -
          3.5320197899716837e-10 * pow(x, 11) + 7.08240282274876e-13 * pow(x, 13) -
          6.053382120104225e-16 * pow(x, 15)) /
         (1. + 0.010116214573922555 * pow(x, 2) + 0.000049917511616975513 * pow(x, 4) +
          1.556549863087456e-7 * pow(x, 6) + 3.280675710557897e-10 * pow(x, 8) +
          4.5049097575386586e-13 * pow(x, 10) + 3.211070511937122e-16 * pow(x, 12));
}

// Pade approximant for Ci(x), with m=12, n=12
inline double Ci_pade_approximant_12_12(double x) {
  using namespace std;
  return log(x) + (0.5772156649015329 - 0.24231497614160186 * pow(x, 2) +
          0.007139183039136621 * pow(x, 4) - 0.00011466618094101764 * pow(x, 6) +
          8.443734405201243e-7 * pow(x, 8) - 3.060472574705558e-9 * pow(x, 10) +
          4.328624073851291e-12 * pow(x, 12)) /
         (1. + 0.013313955815300189 * pow(x, 2) + 0.00008836441800952094 * pow(x, 4) +
          3.800404484365274e-7 * pow(x, 6) + 1.1376490214488613e-9 * pow(x, 8) +
          2.297129602871981e-12 * pow(x, 10) + 2.510407760855278e-15 * pow(x, 12));
}

#endif
