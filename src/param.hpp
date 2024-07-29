/*!
  \file param.hpp
  \author Siyang Ling
  \brief Utilities for managing simulations parameters.
  
  This header file contains utilities for pretty-printing and saving parameters of a simulation.
  By convention, we collect all parameters in a (trivial, standard layout) struct containing double's or long long int's.
  (e.g. SampleParam)
  The utilities here are generic for different parameter structs; 
  you can define your own new type containing new parameters, and use the utilities here as usual.
  Typically, we use these utilities to export a struct along with some meta-information, 
  so that external code (Mathematica / Python) can also use the parameters.
*/
#ifndef PARAM_HPP
#define PARAM_HPP

#include "utility.hpp"
#include "boost/pfr.hpp"
#include "boost/type_index.hpp"
#include <fstream>
#include <string>

/*!
  \brief A sample parameter type specifying a lambda-phi-4 theory in an FRW background.
*/
struct SampleParam {
  long long int N; /*!< the number of lattice points on one side of the box (i.e. \f$ N=256 \f$ means \f$ 256^3 \f$ lattice sites) */
  double L; /*!< the length of one side of the box (i.e. \f$ L=10.0 \f$ means the box has volume \f$ L^3 \f$ ) */
  double m; /*!< mass \f$ m \f$ of the scalar field */
  double lambda; /*!< quartic self-interaction of the scalar field (i.e. \f$ \lambda \f$ in \f$ V(\varphi) = \frac12 m^2 \varphi^2 + \frac14 \lambda \varphi^4 \f$ )   */
  double k_ast; /*!< the wavenumber \f$ k_\ast \f$ for the peak of the field power spectrum */
  double varphi_std_dev; /*!< the expected RMS value \f$ \langle \varphi^2 \rangle \f$ for the field, averaged over the box */
  double a1; /*!< the scale factor at time \f$ t_1 \f$ */
  double H1; /*!< the Hubble parameter at time \f$ t_1 \f$ */
  double t1; /*!< coordinate time parameter \f$ t_1 \f$  (For radiation domination, \f$ a(t) = a_1 (1 + 2 H_1 (t-t_1))^{1/2} \f$, \f$ H(t) = H_1 (1 + 2 H_1 (t-t_1))^{-1} \f$ .) */
};

/*!
  \brief Pretty prints a parameter struct T.
*/
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

/*!
  \brief Save the member names of parameter struct T to filename.
*/
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

namespace {
template<typename T> std::string_view Mathematica_format;

template<> constexpr std::string_view Mathematica_format<double> = "Real64";

template<> constexpr std::string_view Mathematica_format<long long int> = "Integer64";
}

/*!
  \brief Save the member types of parameter struct T to filename. Type names are in Mathematica convention.
*/
template<typename T>
void save_param_Mathematica_formats(const std::string &filename) {
  std::ofstream outstream(filename);
  auto func = [&](const auto &field) {
		typedef std::remove_const_t<std::remove_reference_t<decltype(field)>> type_of_field;
		outstream << Mathematica_format<type_of_field> << '\n';
	      };
  boost::pfr::for_each_field(T(), func);
}

/*!
  \brief Save param directly to filename.
*/
template<typename T>
static void save_param(const T &param, const std::string &filename){
  std::ofstream outstream(filename, std::ios::binary);
  if(outstream.is_open()){
    outstream.write((const char *)&param, sizeof(T));
  }
}

/*!
  \brief Save member names, types and values of param to directory dir.
*/
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
