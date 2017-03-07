#ifndef SGP_SDE_UTILS_H
#define SGP_SDE_UTILS_H

#include <RcppArmadillo.h>

// TODO: split in h + cpp files
inline arma::vec rep_value(double value, int n) {
  arma::vec v(n);
  v.fill(value);
  return v;
}

#endif