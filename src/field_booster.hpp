/*
  Tools related to boosting (adding velocity to) fields.
*/
#ifndef FIELD_BOOSTER_HPP
#define FIELD_BOOSTER_HPP

#include "Eigen/Dense"

void add_phase_to_state(Eigen::VectorXd &state, const Eigen::VectorXd &phase);

void boost_klein_gordon_field_old(Eigen::VectorXd &varphi, Eigen::VectorXd &dt_varphi, const Eigen::VectorXd &theta,
				  const long long int N, const double L, const double m);

Eigen::VectorXd boost_klein_gordon_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, const double abs_delta_t);

Eigen::VectorXd boost_proca_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, const double abs_delta_t);

#endif
