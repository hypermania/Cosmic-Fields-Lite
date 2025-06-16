/*
  Tools related to boosting (adding velocity to) fields.
*/
#ifndef FIELD_BOOSTER_HPP
#define FIELD_BOOSTER_HPP

#include "Eigen/Dense"


Eigen::VectorXd boost_klein_gordon_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, const double abs_delta_t);

Eigen::VectorXd boost_proca_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, Eigen::VectorXd &state_init, const double abs_delta_t, const std::string save_path);

Eigen::ArrayXcd boost_sp_field(const long long int N, const double L, const double m, const Eigen::ArrayXd &tau, const Eigen::ArrayXcd &state_init);

#endif
