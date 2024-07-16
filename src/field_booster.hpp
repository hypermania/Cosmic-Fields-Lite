/*
  Tools related to boosting (adding velocity to) fields.
*/
#ifndef FIELD_BOOSTER_HPP
#define FIELD_BOOSTER_HPP

#include <array>
#include <deque>
#include <boost/math/interpolators/quintic_hermite.hpp>

#include "workspace.hpp"
#include "equations.hpp"
#include "fdm3d.hpp"
#include "utility.hpp"



void add_phase_to_state(Eigen::VectorXd &state, const Eigen::VectorXd &phase);

void boost_klein_gordon_field(Eigen::VectorXd &varphi, Eigen::VectorXd &dt_varphi, const Eigen::VectorXd &theta,
			      const long long int N, const double L, const double m);


#endif
