#include "io.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void assert_near(const double actual, const double expected)
{
  assert(std::abs(actual - expected) < 1e-12);
}

} // namespace

int main()
{
  const std::filesystem::path dir = "output/io_test";
  std::filesystem::create_directories(dir);

  const std::vector<double> std_vector{1.25, -2.5, 3.75};
  const std::string vector_path = (dir / "std_vector.dat").string();
  write_to_file(std_vector, vector_path);
  const std::vector<double> loaded_std_vector = load_vector_from_file(vector_path);
  assert(loaded_std_vector.size() == std_vector.size());
  for(std::size_t i = 0; i < std_vector.size(); ++i) {
    assert_near(loaded_std_vector[i], std_vector[i]);
  }

  Eigen::VectorXd eigen_vector(3);
  eigen_vector << -4.0, 5.5, 6.25;
  const std::string eigen_path = (dir / "eigen_vector.dat").string();
  write_to_file(eigen_vector, eigen_path);
  const Eigen::VectorXd loaded_eigen_vector = load_VectorXd_from_file(eigen_path);
  assert(loaded_eigen_vector.size() == eigen_vector.size());
  for(Eigen::Index i = 0; i < eigen_vector.size(); ++i) {
    assert_near(loaded_eigen_vector[i], eigen_vector[i]);
  }

  const std::string long_prefix(160, 'x');
  const std::string templated_path = (dir / (long_prefix + "_%d.dat")).string();
  write_to_filename_template(eigen_vector, templated_path, 42);
  assert(std::filesystem::exists(dir / (long_prefix + "_42.dat")));

  bool missing_file_threw = false;
  try {
    (void)load_VectorXd_from_file((dir / "missing.dat").string());
  } catch(const std::runtime_error &) {
    missing_file_threw = true;
  }
  assert(missing_file_threw);

  return 0;
}
