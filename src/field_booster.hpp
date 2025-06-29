/*!
  \file field_booster.hpp
  \author Siyang Ling
  \brief Tools and examples for spatially varying boost of fields.
*/
#ifndef FIELD_BOOSTER_HPP
#define FIELD_BOOSTER_HPP

#include "Eigen/Dense"

/*!
  \brief Boost Klein Gordon field state_init with (non-relativistic) time slice function tau.
*/
Eigen::VectorXd boost_klein_gordon_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, const Eigen::VectorXd &state_init, const double abs_delta_t);

/*!
  \brief Boost Proca field state_init with (non-relativistic) time slice function tau.
*/
Eigen::VectorXd boost_proca_field(const long long int N, const double L, const double m, const Eigen::VectorXd &tau, Eigen::VectorXd &state_init, const double abs_delta_t, const std::string save_path);

/*!
  \brief Boost Schroedinger Poisson field state_init with (non-relativistic) time slice function tau.
*/
Eigen::ArrayXcd boost_sp_field(const long long int N, const double L, const double m, const Eigen::ArrayXd &tau, const Eigen::ArrayXcd &state_init);

/*!
  \brief An example for boosting of Klein Gordon field.
*/
void generate_ic_kg(void);

/*!
  \brief An example for boosting of Proca field.  See spatially varying boost paper.
*/
void generate_ic_proca(void);

/*!
  \brief An example for boosting of Shroedinger Poisson field.  See spatially varying boost paper.
*/
void generate_ic_sp(void);

/*!
  \brief An example for boosting of Sine Gordon field. See spatially varying boost paper.
*/
void generate_ic_sg(void);


#endif
