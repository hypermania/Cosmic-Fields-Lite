#ifndef IO_HPP
#define IO_HPP
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

#include <Eigen/Dense>


std::vector<double> load_vector_from_file(std::string filename);
void write_vector_to_file(std::vector<double> vector, std::string filename);
void write_data_to_file(const char *buf, ssize_t size, std::string filename);

void write_VectorXd_to_file(const Eigen::VectorXd &vector, std::string filename);
void write_VectorXd_to_filename_template(const Eigen::VectorXd &vector, const std::string format_string, const int idx);
Eigen::VectorXd load_VectorXd_from_file(const std::string &filename);


#endif
