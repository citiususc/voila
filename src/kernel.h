#ifndef GPSDE_KERNEL_H
#define GPSDE_KERNEL_H

#include <RcppArmadillo.h>
#include <functional>

class kernel {
public:
  typedef std::function<double (const arma::vec&,
                                const arma::vec&,
                                const arma::vec&)> kernel_expression;
  kernel(int inputDimension, const arma::vec& hyperparams,
         const kernel_expression& expression,
         const arma::vec& lowerBound, const arma::vec& upperBound,
         double epsilon = 0.0);
  kernel(int inputDimension,
         const arma::vec& hyperparams,
         const kernel_expression& expression,
         double epsilon);
  virtual ~kernel();
  void set_hyperparams(const arma::vec& hyperparams);
  arma::vec get_hyperparams() const;
  void increase_lower_bound (arma::vec& lowerBound);
  arma::vec get_lower_bound();
  void decrease_upper_bound (arma::vec& upperBound);
  arma::vec get_upper_bound();
  double get_epsilon() const;
  void set_epsilon(double epsilon);
  arma::mat covmat(const arma::mat& x, const arma::mat& y);
  arma::mat autocovmat(const arma::mat& x);
  arma::vec variances(const arma::mat& x);
  kernel sum_kernel(const kernel& ker);
  kernel multiply_kernel(const kernel& ker);
  kernel scale_kernel(double constant);

protected:
  int mInputDimension;
  arma::vec mHP;
  std::function<double (const arma::vec&, const arma::vec&,const arma::vec&)> mExpression;
  arma::vec mLowerBound;
  arma::vec mUpperBound;
  double mEpsilon;

  static void sanitize_input_dimension(int vectorDimension);
  static void sanitize_hyperparams(const arma::vec& hyperparams,
                                   const arma::vec& lowerBound,
                                   const arma::vec& upperBound);
  static void check_hyperparameters_within_bounds(const arma::vec& hyperparams,
                                                  const arma::vec& lowerBound,
                                                  const arma::vec& upperBound);
  static void check_bounds_consistency(const arma::vec& lowerBound,
                                       const arma::vec& upperBound);
  static void check_epsilon(double epsilon);
  void check_input_dimension(int vectorDimension);

};

#endif
