/*! 
  \file io.hpp
  \author Siyang Ling
  \brief Input/output utilities.
*/
#ifndef IO_HPP
#define IO_HPP
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

#include <Eigen/Dense>


Eigen::VectorXd load_VectorXd_from_file(const std::string &filename);


template<typename Scalar>
void write_to_file(const std::vector<Scalar> &vector, std::string filename){
  const char *memblock = reinterpret_cast<const char *>(vector.data());
  std::ofstream file(filename, std::ios::binary);
  if(file.is_open()){
    file.write(memblock, vector.size() * sizeof(Scalar));
  }
}

template<typename Derived>
void write_to_file(const Eigen::PlainObjectBase<Derived> &obj, std::string filename){
  std::ofstream file(filename, std::ios::binary);
  if(file.is_open()){
    file.write((char *)obj.data(), obj.size() * sizeof(typename Eigen::DenseBase<Derived>::Scalar));
  }
}

template<typename Derived>
void write_to_filename_template(const Eigen::PlainObjectBase<Derived> &obj, const std::string format_string, const int idx)
{
  char filename[128];
  sprintf(filename, format_string.data(), idx);
  std::ofstream file(filename, std::ios::binary);
  if(file.is_open()){
    file.write((char *)obj.data(), obj.size() * sizeof(typename Eigen::DenseBase<Derived>::Scalar));
  }
}


#endif
