/*
  Settings on numerical tolerance.
*/
#ifndef TOLERANCE_H
#define TOLERANCE_H

/*
  See "Automatic Step Size Control" (pp.167, Chapter II.4) 
  in "Solving Ordinary Differential Equations I" by Hairer, Norsett, Wanner
  for a discussion of absolute and relative tolerance.
 */
constexpr static double ABS_TOL = 1.0e-9;
constexpr static double REL_TOL = 1.0e-9;

#endif
