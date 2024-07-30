/*!
  \file utility.hpp
  \author Siyang Ling
  \brief Utilities for debugging / profiling / pretty printing.
*/
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

// Pretty print functions
inline static std::string line_separator_with_description(const std::string &description) {
  std::string result(80, '=');
  const int length = description.length() + 2;
  result.replace(80 / 2 - length / 2, length, " " + description + " ");
  return result;
}


inline static std::string line_separator_with_description(void) {
  std::string result(80, '=');
  return result;
}


template<typename Callable>
static void run_and_print(const std::string &description, const Callable &c) {
  std::cout << line_separator_with_description(description) << '\n';
  c();
  std::cout << line_separator_with_description() << '\n';
}


template<typename Callable>
static void run_and_measure_time(const std::string &description, const Callable &c) {
  std::cout << line_separator_with_description(description) << '\n';
  auto time_start = std::chrono::system_clock::now();
  c();
  auto time_end = std::chrono::system_clock::now();
  std::chrono::duration<double> time_diff = time_end - time_start;
  std::cout << std::fixed << std::setprecision(9) << std::left;
  std::cout << std::setw(9) << "time spent = " << time_diff.count() << " s" << '\n';
  std::cout << line_separator_with_description() << '\n';
}


static void prepare_directory_for_output(const std::string &dir) {
  const std::filesystem::path dir_path(dir);
  std::error_code ec;
  std::cout << line_separator_with_description("Preparing directory for output") << '\n';
  std::cout << "Saving results in directory: " << dir << '\n';
  std::filesystem::create_directories(dir_path, ec);
  std::cout << "ErrorCode = " << ec.message() << '\n';
  std::cout << line_separator_with_description() << '\n';
}


// Simple profiler for a big task, taking many cycles
// Note that the function call incurs some time cost, so this is not totally accurate
template<typename Callable> 
inline void profile_function(long long int repeat, Callable &&c) {
  auto time_start = std::chrono::system_clock::now();
  for(long long int i = 0; i < repeat; ++i) {
    c();
  }
  std::cout << line_separator_with_description("Profiling a callable") << '\n';
  auto time_end = std::chrono::system_clock::now();
  std::chrono::duration<double> time_diff = time_end - time_start;
  std::cout << std::fixed << std::setprecision(9) << std::left;
  std::cout << std::setw(9) << "total time spent = " << time_diff.count() << " s" << '\n';
  std::cout << std::setw(9) << "time spent per iteration = " << time_diff.count() / repeat << " s" << '\n';
  std::cout << line_separator_with_description() << '\n';
}



#endif
