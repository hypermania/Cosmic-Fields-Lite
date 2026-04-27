#include "io.hpp"

namespace {

std::streamsize binary_file_size(std::ifstream &file, const std::string &filename)
{
  const std::streampos size = file.tellg();
  if(size < 0) {
    throw std::runtime_error("Unable to determine file size: " + filename);
  }
  if(static_cast<std::streamoff>(size) % static_cast<std::streamoff>(sizeof(double)) != 0) {
    throw std::runtime_error("Binary file size is not a multiple of sizeof(double): " + filename);
  }
  return static_cast<std::streamsize>(size);
}

void read_binary_doubles(std::ifstream &file, char *data, const std::streamsize size, const std::string &filename)
{
  file.seekg(0, std::ios::beg);
  if(size > 0) {
    file.read(data, size);
  }
  if(!file) {
    throw std::runtime_error("Failed to read file: " + filename);
  }
}

} // namespace

std::vector<double> load_vector_from_file(const std::string &filename){
  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if(!file.is_open()){
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }

  const std::streamsize size = binary_file_size(file, filename);
  std::vector<double> v(static_cast<std::size_t>(size) / sizeof(double));
  read_binary_doubles(file, reinterpret_cast<char *>(v.data()), size, filename);

  return v;
}

Eigen::VectorXd load_VectorXd_from_file(const std::string &filename){
  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if(!file.is_open()){
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }

  const std::streamsize size = binary_file_size(file, filename);
  Eigen::VectorXd v(static_cast<Eigen::Index>(size / sizeof(double)));
  read_binary_doubles(file, reinterpret_cast<char *>(v.data()), size, filename);
  
  return v;
}
