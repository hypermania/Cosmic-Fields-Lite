#include "io.hpp"

std::vector<double> load_vector_from_file(std::string filename){
  std::streampos size = 0;
  char *memblock = 0;

  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if(file.is_open()){
    size = file.tellg();
    memblock = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(memblock, size);
  }
  
  file.close();
  
  //std::cout << "Loading " << filename << ". Size of file is " << size << " bytes.\n";

  double *double_values = (double *)memblock;
  unsigned long long int N = (unsigned long long int)size / sizeof(double);
  
  std::vector<double> v(N);
  for(unsigned long long int i = 0; i < N; i++){
    v[i] = double_values[i];
  }

  return v;
}

Eigen::VectorXd load_VectorXd_from_file(const std::string &filename){
  std::streampos size = 0;

  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if(file.is_open()){
    size = file.tellg();
    file.seekg(0, std::ios::beg);
  }

  unsigned long long int N = (unsigned long long int)size / sizeof(double);
  Eigen::VectorXd v(N);
  
  file.read((char *)v.data(), size);
  
  return v;
}

