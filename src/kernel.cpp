#include "kernel.h"
using namespace Rcpp;

//TODO
kernel::kernel(int inputDimension): kernel(inputDimension, arma::vec({}),
               [](const arma::vec&, const arma::vec&, const arma::vec&) -> double {
                 return 0;
               }, arma::vec({}), arma::vec({}), 0) {
}

kernel::kernel(int inputDimension, const arma::vec& hyperparams,
       const std::function<double(const arma::vec&, const arma::vec&,const arma::vec&)>& expression,
       const arma::vec& lowerBound, const arma::vec& upperBound, double epsilon)
       // note the use of the , operator to sanitize the inputs
       : mInputDimension( (kernel::sanitize_input_dimension(inputDimension), inputDimension) ),
         mHP( (kernel::sanitize_hyperparams(hyperparams, lowerBound, upperBound),
               hyperparams) ),
         mExpression(expression),
         mLowerBound(lowerBound),
         mUpperBound(upperBound),
         mEpsilon( (kernel::check_epsilon(epsilon), epsilon) ){
  }


kernel::kernel(int inputDimension, const arma::vec& hyperparams,
               const std::function<double(const arma::vec&, const arma::vec&, const arma::vec&)>& expression,
               double epsilon)
  : mInputDimension( (kernel::sanitize_input_dimension(inputDimension), inputDimension)),
    mHP(hyperparams), mLowerBound(hyperparams.n_elem), mUpperBound(hyperparams.n_elem),
    mExpression(expression),
    mEpsilon( (kernel::check_epsilon(epsilon), epsilon) ) {
  mLowerBound.fill(0);
  mUpperBound.fill(std::numeric_limits<double>::max());
  check_hyperparameters_within_bounds(mHP, mLowerBound, mUpperBound);
}


kernel::~kernel() {
}

int kernel::get_input_dimension() const {
  return mInputDimension;
}

void kernel::set_hyperparams(const arma::vec& hyperparams) {
  if ((hyperparams.n_elem != mHP.n_elem)) {
    throw std::invalid_argument("Invalid hyperparams' length");
  }
  if (arma::any(hyperparams > mUpperBound) ||
      arma::any(hyperparams < mLowerBound)) {
    throw std::invalid_argument("hyperparams is not contained withing \
                                [lowerBound, upperBound");
  }
  mHP = hyperparams;
}


arma::vec kernel::get_hyperparams() const {
  return mHP;
}

// the lower bound only can be increased with respect to the initial value to
// ensure correctness
void kernel::increase_lower_bound(arma::vec& lowerBound){
  check_bounds_consistency(lowerBound, mUpperBound);
  if (arma::any(lowerBound < mLowerBound)) {
    throw std::invalid_argument("The lower bound can only be increased");
  }
  mLowerBound = lowerBound;
}


arma::vec kernel::get_lower_bound() const {
  return mLowerBound;
}


// the upper bound only can be decreased with respect to the initial value to
// ensure correctness
void kernel::decrease_upper_bound (arma::vec& upperBound){
  check_bounds_consistency(mLowerBound, upperBound);
  if (arma::any(upperBound > mUpperBound)) {
    throw std::invalid_argument("The upper bound can only be decreased");
  }
  mUpperBound = upperBound;
}


arma::vec kernel::get_upper_bound() const {
  return mUpperBound;
}

double kernel::get_epsilon() const {
  return mEpsilon;
}

void kernel::set_epsilon(double epsilon) {
  check_epsilon(epsilon);
  mEpsilon = epsilon;
}


arma::mat kernel::covmat(const arma::mat& x,
                         const arma::mat& y) const {
  int N = x.n_rows;
  int M = y.n_rows;
  int D = x.n_cols;

  if (D != y.n_cols) {
    throw std::invalid_argument("The dimensions of the vectors do not match");
  }

  check_input_dimension(D);

  arma::mat cov(N,M);
  arma::vec xi, xj;
  for (int i = 0; i < N; i++) {
    xi = arma::conv_to< arma::vec >::from(x.row(i));
    for (int j = 0; j < M; j++) {
      xj = arma::conv_to< arma::vec >::from(y.row(j));
      cov(i,j) = mExpression(xi, xj, mHP);
    }
  }
  // TODO: epsilon with covmat?
  return cov;
}

arma::mat kernel::autocovmat(const arma::mat& x) const {
  int N = x.n_rows;
  check_input_dimension(x.n_cols);

  arma::mat cov(N,N);
  arma::vec xi, xj;
  for (int i = 0; i < N; i++) {
    xi = arma::conv_to< arma::vec >::from(x.row(i));
    cov(i,i) = mExpression(xi,xi, mHP);
    for (int j = i + 1; j < N; j++) {
      xj = arma::conv_to< arma::vec >::from(x.row(j));
      cov(i,j) = mExpression(xi, xj, mHP);
      cov(j,i) = cov(i,j);
    }
  }
  // Add small fact or for better numerical stability
  if (mEpsilon != 0) {
    for (int i = 0; i < N; i++) {
      cov(i,i) += mEpsilon;
    }
  }
  return cov;
}

arma::vec kernel::variances(const arma::mat& x) const {
  int N = x.n_rows;

  check_input_dimension(x.n_cols);

  arma::colvec cov(N);
  arma::vec xi;
  for (int i = 0; i < N; i++) {
    xi = arma::conv_to< arma::vec >::from(x.row(i));
    cov(i) = mExpression(xi, xi, mHP);
  }
  // Add small fact or for consistency with autocov
  if (mEpsilon != 0) {
    for (int i = 0; i < N; i++) {
      cov(i) += mEpsilon;
    }
  }
  return cov;

}

kernel kernel::sum_kernel(const kernel& ker) {
  if (this->mInputDimension != ker.mInputDimension) {
    throw std::invalid_argument("The input Dimension of the Kernels doest not \
                                 match");
  }
  auto f1 = this->mExpression, f2 = ker.mExpression;
  int l1 = this->mHP.n_elem;
  int l2 = (ker.mHP).n_elem;
  std::function<double(const arma::vec&, const arma::vec&,
                       const arma::vec&)> exp =
                         [f1, f2, l1, l2](const arma::vec& x, const arma::vec& y,
                                          const arma::vec& hp) -> double {
                                            arma::vec hp1, hp2;
                                            if (l1 > 0) {
                                              hp1 =  hp.subvec(0, l1 - 1);
                                            }
                                            if (l2 > 0) {
                                              hp2 =  hp.subvec(l1, l1 + l2 - 1);
                                            }
                                            return f1(x, y, hp1) + f2(x, y, hp2);
                                          };
  return kernel(this->mInputDimension,
                arma::join_cols(this->mHP, ker.mHP), exp,
                arma::join_cols(this->mLowerBound, ker.mLowerBound),
                arma::join_cols(this->mUpperBound, ker.mUpperBound),
                this->mEpsilon + ker.mEpsilon);
}


kernel kernel::multiply_kernel(const kernel& ker) {
  if (this->mInputDimension != ker.mInputDimension) {
    throw std::invalid_argument("The input Dimension of the Kernels doest not \
                                  match");
  }
  auto f1 = this->mExpression, f2 = ker.mExpression;
  int l1 = this->mHP.n_elem;
  int l2 = (ker.mHP).n_elem;
  std::function<double(const arma::vec&, const arma::vec&,
                       const arma::vec&)> exp =
                         [f1, f2, l1, l2](const arma::vec& x, const arma::vec& y,
                                          const arma::vec& hp) -> double {
                                            arma::vec hp1, hp2;
                                            if (l1 > 0) {
                                              hp1 =  hp.subvec(0, l1 - 1);
                                            }
                                            if (l2 > 0) {
                                              hp2 =  hp.subvec(l1, l1 + l2 - 1);
                                            }
                                            return f1(x, y, hp1) * f2(x, y, hp2);
                                          };
  return kernel(this->mInputDimension,
                arma::join_cols(this->mHP, ker.mHP), exp,
                arma::join_cols(this->mLowerBound, ker.mLowerBound),
                arma::join_cols(this->mUpperBound, ker.mUpperBound),
                this->mEpsilon + ker.mEpsilon);
}

// constant * kernel
kernel kernel::scale_kernel(double constant) {
  if (constant < 0) {
    throw std::invalid_argument("Constant should be >= 0");
  }
  auto f1 = this->mExpression;
  int l1 = this->mHP.n_elem;
  std::function<double(const arma::vec&, const arma::vec&,
                       const arma::vec&)> exp =
                         [f1, constant](const arma::vec& x, const arma::vec& y,
                              const arma::vec& hp) -> double {
                                            return constant * f1(x, y, hp);
                              };
  return kernel(this->mInputDimension,
                this->mHP, exp,
                this->mLowerBound,
                this->mUpperBound,
                this->mEpsilon);
}


void kernel::sanitize_input_dimension(int vectorDimension) {
  if (vectorDimension < 1) {
    throw std::invalid_argument("Input dimension is smaller than 1");
  }
}

void kernel::sanitize_hyperparams(const arma::vec& hyperparams,
                                  const arma::vec& lowerBound,
                                  const arma::vec& upperBound) {
  if (lowerBound.n_elem != hyperparams.n_elem) {
    throw std::invalid_argument("Size of lower bound differs from size of \
                                 hyperparameters");
  }
  if (upperBound.n_elem != hyperparams.n_elem) {
    throw std::invalid_argument("Size of upper bound differs from size of \
                                 hyperparameters");
  }
  check_bounds_consistency(lowerBound, upperBound);
  check_hyperparameters_within_bounds(hyperparams, lowerBound, upperBound);
}


void kernel::check_hyperparameters_within_bounds(const arma::vec& hyperparams,
                                                 const arma::vec& lowerBound,
                                                 const arma::vec& upperBound) {
  if (arma::any(hyperparams > upperBound) || (arma::any(hyperparams < lowerBound))) {
    throw std::invalid_argument("The hyperparameters are not contained within \
                                 [lowerBound, upperBound]");
  }
}


void kernel::check_bounds_consistency(const arma::vec& lowerBound,
                                      const arma::vec& upperBound) {
  if (arma::any(lowerBound > upperBound)) {
    throw std::invalid_argument("Some values of the upper bound are \
                                 below the lower bound");
  }
}


void kernel::check_epsilon(double epsilon) {
  if (epsilon < 0) {
    throw std::invalid_argument("Epsilon should be >= 0");
  }
}

void kernel::check_input_dimension(int vectorDimension) const {
  if (vectorDimension != mInputDimension) {
    throw std::invalid_argument("The input Dimension of the Kernel doest not \
                                   match the dimension of the vectors");
  }
}
