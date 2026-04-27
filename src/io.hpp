/*! 
  \file io.hpp
  \author Siyang Ling
  \brief Input/output utilities.
*/
#ifndef IO_HPP
#define IO_HPP
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>

#include <Eigen/Dense>


Eigen::VectorXd load_VectorXd_from_file(const std::string &filename);
std::vector<double> load_vector_from_file(const std::string &filename);


template<typename Scalar>
void write_to_file(const std::vector<Scalar> &vector, const std::string &filename){
  const char *memblock = reinterpret_cast<const char *>(vector.data());
  std::ofstream file(filename, std::ios::binary);
  if(!file.is_open()){
    throw std::runtime_error("Unable to open file for writing: " + filename);
  }
  file.write(memblock, vector.size() * sizeof(Scalar));
  if(!file) {
    throw std::runtime_error("Failed to write file: " + filename);
  }
}

template<typename Derived>
void write_to_file(const Eigen::PlainObjectBase<Derived> &obj, const std::string &filename){
  std::ofstream file(filename, std::ios::binary);
  if(!file.is_open()){
    throw std::runtime_error("Unable to open file for writing: " + filename);
  }
  file.write(reinterpret_cast<const char *>(obj.data()), obj.size() * sizeof(typename Eigen::DenseBase<Derived>::Scalar));
  if(!file) {
    throw std::runtime_error("Failed to write file: " + filename);
  }
}

template<typename Derived>
void write_to_filename_template(const Eigen::PlainObjectBase<Derived> &obj, const std::string &format_string, const int idx)
{
  const int length = std::snprintf(nullptr, 0, format_string.c_str(), idx);
  if(length < 0) {
    throw std::runtime_error("Failed to format output filename: " + format_string);
  }
  std::vector<char> filename(length + 1);
  std::snprintf(filename.data(), filename.size(), format_string.c_str(), idx);
  write_to_file(obj, std::string(filename.data(), length));
}


#endif
