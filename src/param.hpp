/*!
  \file param.hpp
  \brief Utilities for managing parameters specifying different kinds of simulations.
  
  Data types and utilities for managing the parameters of a simulation.
  The parameters are collected as a struct of double or long long int.
  Uses Boost.PFR for saving struct member type names, 
  so that Mathematica can import the struct in binary.
*/
#ifndef PARAM_HPP
#define PARAM_HPP

#include "utility.hpp"
#include "boost/pfr.hpp"
#include "boost/type_index.hpp"
#include <fstream>
#include <iostream>
#include <string>

/*
  Utilities for managing parameters specifying different kinds of simulations.
  SampleParam is a parameter type specifying a lambda-phi-4 theory in an FRW background.
  You may define your own parameter type, but preferably using conventions consistent with below:

  Discretization parameters:
  N: the number of lattice points on one side of the box (i.e. N=256 means 256^3 lattice sites)
  L: the length of one side of the box (i.e. L=10.0 means the box has volume L^3)

  ULDM field parameters:
  m: mass of the ULDM field
  lambda: quartic self-interaction of the ULDM field 
  (i.e. V(varphi) = m^2 varphi^2 / 2 + lambda varphi^4 / 4)
  f_a: the "nonlinear scale" of the ULDM field
  (i.e. V(varphi) = m^2 f_a^2 (sqrt(1 + varphi^2 / f_a^2) - 1))
  k_ast: the wavenumber for the peak of the field power spectrum
  k_Psi: the wavenumber for a large scale perturbation in the field energy density
  varphi_std_dev: the expected RMS value <varphi^2> for the field, averaged over the box
  Psi_std_dev: the expected RMS value <Psi^2> for the perturbation, averaged over the box

  Cosmological parameters:
  a1: the scale factor at time t1  
  H1: the Hubble parameter at time t1
  t1: coordinate time parameter
  (For radiation domination, a(t) = a1 * (1 + 2 H1 (t-t1))^0.5, H(t) = H1 * (1 + 2 H1 (t-t1))^(-1) .)
*/

struct SampleParam {
  long long int N;
  double L;
  double m;
  double lambda;
  double k_ast;
  double k_Psi;
  double varphi_std_dev;
  double Psi_std_dev;
  double a1;
  double H1;
  double t1;
};


template<typename T>
void print_param(const T &param) {
  auto names = boost::pfr::names_as_array<T>();
  auto func = [&](const auto &field, std::size_t i) {
		std::cout << names[i] << ": " << field
			  << " (" << boost::typeindex::type_id_runtime(field) << ")\n";
	      };
  // std::cout << line_separator_with_description("The parameters for the simulation") << '\n';
  // boost::pfr::for_each_field(param, func);
  // std::cout << line_separator_with_description() << '\n';
  auto c = [&](){ boost::pfr::for_each_field(param, func); };
  run_and_print("The parameters for the simulation", c);
}


template<typename T>
void save_param_names(const std::string &filename) {
  std::ofstream outstream(filename);
  auto names = boost::pfr::names_as_array<T>();
  for(auto name : names) {
    outstream << name << '\n';
  }
}

/*
// Compiles with Intel icpx, but doesn't compile with gcc due to "Explicit template specialization cannot have a storage class"
template<typename T> std::string_view Mathematica_format;

template<> constexpr static std::string_view Mathematica_format<double> = "Real64";

template<> constexpr static std::string_view Mathematica_format<long long int> = "Integer64";
*/

/*
// Compiles with gcc, fails at link stage with Intel icpx due to multiple definitions
template<typename T> std::string_view Mathematica_format;

template<> constexpr std::string_view Mathematica_format<double> = "Real64";

template<> constexpr std::string_view Mathematica_format<long long int> = "Integer64";
*/

/*
// Alternate solution
constexpr static std::string get_Mathematica_format(double) {
  return std::string("Real64");
}

constexpr static std::string get_Mathematica_format(long long int) {
  return std::string("Integer64");
}
*/

namespace {
template<typename T> std::string_view Mathematica_format;

template<> constexpr std::string_view Mathematica_format<double> = "Real64";

template<> constexpr std::string_view Mathematica_format<long long int> = "Integer64";
}

template<typename T>
void save_param_Mathematica_formats(const std::string &filename) {
  std::ofstream outstream(filename);
  auto func = [&](const auto &field) {
		typedef std::remove_const_t<std::remove_reference_t<decltype(field)>> type_of_field;
		outstream << Mathematica_format<type_of_field> << '\n';
	      };
  boost::pfr::for_each_field(T(), func);
}


template<typename T>
static void save_param(const T &param, const std::string &filename){
  std::ofstream outstream(filename, std::ios::binary);
  if(outstream.is_open()){
    outstream.write((const char *)&param, sizeof(T));
  }
}


template<typename T>
void save_param_for_Mathematica(const T &param, const std::string &dir) {
  save_param_names<T>(dir + "paramNames.txt");
  save_param_Mathematica_formats<T>(dir + "paramTypes.txt");
  save_param<T>(param, dir + "param.dat");
}


template<typename T>
void save_param_types(const std::string &filename) {
  std::ofstream outstream(filename);
  auto func = [&](const auto &field) {
		outstream << boost::typeindex::type_id_runtime(field) << '\n';
	      };
  boost::pfr::for_each_field(T(), func);
}




#endif
