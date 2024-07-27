/*
  Tools related to boosting (adding velocity to) fields.
*/
#ifndef FIELD_BOOSTER_HPP
#define FIELD_BOOSTER_HPP

#include "Eigen/Dense"

void add_phase_to_state(Eigen::VectorXd &state, const Eigen::VectorXd &phase);

void boost_klein_gordon_field(Eigen::VectorXd &varphi, Eigen::VectorXd &dt_varphi, const Eigen::VectorXd &theta,
			      const long long int N, const double L, const double m);


#endif
