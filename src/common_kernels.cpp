#include "common_kernels.h"
#include "utils.h"
#include <cmath>
#include <cassert>
using namespace Rcpp;

// TODO:: additional checks depending on kernel type
// TODO:: CHANGE ASSERTS

// exponential kernel
exponential_kernel::exponential_kernel(int inputDimension,
                                       double amplitude,
                                       const arma::vec& lengthScales,
                                       double epsilon)
  : kernel( (check_length_scales_dims(inputDimension, lengthScales), inputDimension),
             lengthScales,
             exponential_kernel::generate_exponential_kernel(amplitude), epsilon)  {
}

kernel::kernel_expression exponential_kernel::generate_exponential_kernel(double amplitude) {
  kernel::kernel_expression exponential =
    [amplitude](const arma::vec& x, const arma::vec&y,
       const arma::vec& hyperparams) -> double {
         return amplitude *
           std::exp(- arma::accu( arma::square((x - y) / hyperparams) ) / 2.0);
       };
  return exponential;
}


// rational quadratic kernel
rational_quadratic_kernel::rational_quadratic_kernel(int inputDimension,
                                                     double amplitude,
                                                     double alpha,
                                                     double lengthScale,
                                                     double epsilon)
  : kernel(inputDimension,
           arma::vec({alpha, lengthScale}),
           rational_quadratic_kernel::generate_rational_quadratic_kernel(amplitude),
           epsilon)  {
  }

kernel::kernel_expression rational_quadratic_kernel::generate_rational_quadratic_kernel(double amplitude) {
  kernel::kernel_expression rational_quadratic =
    [amplitude](const arma::vec& x, const arma::vec&y,
       const arma::vec& hyperparams) -> double {
         return amplitude *
           std::pow(1 + std::pow(arma::norm(x - y),2) /
           (2 * hyperparams(0) * hyperparams(1) *  hyperparams(1) ),
           -hyperparams(0));
       };
  return rational_quadratic;
}

sum_exponential_kernels::sum_exponential_kernels(int inputDimension,
                                                 double completeAmplitude,
                                                 double A1,
                                                 const arma::vec& lengthScales1,
                                                 const arma::vec& lengthScales2,
                                                 double epsilon)
  : kernel( (check_length_scales_dims(inputDimension, lengthScales1),inputDimension) ,
    (assert(A1  <= completeAmplitude),
     arma::join_cols(arma::vec{A1}, arma::join_cols(lengthScales1,lengthScales2)) ),
     sum_exponential_kernels::generate_sum_exponential_kernels(completeAmplitude),
     arma::join_cols(arma::vec({0}),rep_value(0, 2 * inputDimension)),
     arma::join_cols(arma::vec({completeAmplitude}),
                     rep_value(std::numeric_limits<double>::max(),
                               2 * inputDimension)),
                               epsilon) {
}

kernel::kernel_expression sum_exponential_kernels::generate_sum_exponential_kernels(double amplitude){
  kernel::kernel_expression exponential =
    [amplitude](const arma::vec& x, const arma::vec&y,
                const arma::vec& hyperparams) -> double {
                  int n = x.n_elem;
                  return hyperparams(0) *
                    std::exp(- arma::accu( arma::square((x - y) / hyperparams.subvec(1, n)) ) / 2.0) +
                         (amplitude - hyperparams(0)) *
                    std::exp(- arma::accu( arma::square((x - y) / hyperparams.subvec(n + 1, hyperparams.n_elem - 1)) ) / 2.0);
                };
    return exponential;
}


exponential_constant_kernel::exponential_constant_kernel(int inputDimension,
                                                         double completeAmplitude,
                                                         double expAmplitude,
                                                         const arma::vec& lengthScales,
                                                         double epsilon)
  : kernel( (check_length_scales_dims(inputDimension, lengthScales),inputDimension) ,
    (assert(expAmplitude  <= completeAmplitude),
     arma::join_cols(arma::vec{expAmplitude}, lengthScales) ),
     exponential_constant_kernel::generate_expression(completeAmplitude),
     arma::join_cols(arma::vec({0}),rep_value(0, inputDimension)),
     arma::join_cols(arma::vec({completeAmplitude}),
                     rep_value(std::numeric_limits<double>::max(),
                               inputDimension)),
                               epsilon)  {
}


kernel::kernel_expression exponential_constant_kernel::generate_expression(double completeAmplitude) {
  kernel::kernel_expression tied_exponential_linear =
    [completeAmplitude](const arma::vec& x, const arma::vec&y,
                        const arma::vec& hyperparams) -> double {
                          return (
                              hyperparams(0) * std::exp(-arma::accu(
                                  arma::square((x - y) / hyperparams.subvec(1, hyperparams.n_elem - 1))
                              ) / 2.0) + (completeAmplitude - hyperparams(0))
                          );
                        };
    return tied_exponential_linear;
}




