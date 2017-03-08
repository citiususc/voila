#ifndef SGP_SDE_UTILS_H
#define SGP_SDE_UTILS_H

#include <RcppArmadillo.h>

// TODO: split in h + cpp files
inline arma::vec rep_value(double value, int n) {
  arma::vec v(n);
  v.fill(value);
  return v;
}


inline arma::mat vector_to_matrix(const arma::vec& x, int nrow, int ncol){
  arma::mat m;
  m.insert_cols(0, x);
  m.reshape(nrow, ncol);
  return m;
}

inline arma::vec matrix_to_vector(const arma::mat& x) {
  return arma::vectorise(x);
}

#endif