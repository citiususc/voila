#include <RcppArmadillo.h>
#include "kernel.h"
#include "common_kernels.h"
#include "sde_variational_inferencer.h"

using namespace Rcpp;

List do_sde_inference(int timeSeriesIndex,
                      const arma::mat& timeSeries,
                      double samplingPeriod,
                      arma::mat& xm,
                      kernel& fKernel,
                      kernel& sKernel, double v,
                      int maxIterations = 20, int verboseLevel = 1) {
  // TODO: add parameters for setting algorithm parameters
  sde_variational_inferencer sdeVI(timeSeries, samplingPeriod);
  sdeVI.set_max_iterations(maxIterations);
  sdeVI.set_verbose_level(verboseLevel);
  return sdeVI.do_inference(timeSeriesIndex, xm, fKernel, sKernel, v);

}


RCPP_EXPOSED_CLASS(kernel);
RCPP_EXPOSED_CLASS(exponential_kernel);
RCPP_EXPOSED_CLASS(rational_quadratic_kernel);
RCPP_EXPOSED_CLASS(exponential_constant_kernel);
RCPP_EXPOSED_CLASS(sum_exponential_kernels);

RCPP_MODULE(KERNELS){
  class_<kernel>("gp_kernel")
  .method("get_hyperparams", &kernel::get_hyperparams)
  .method("set_hyperparams", &kernel::set_hyperparams)
  .method("get_lower_bound", &kernel::get_lower_bound)
  .method("increase_lower_bound", &kernel::increase_lower_bound)
  .method("get_upper_bound", &kernel::get_upper_bound)
  .method("decrease_upper_bound", &kernel::decrease_upper_bound)
  .method("covmat", &kernel::covmat)
  .method("autocovmat", &kernel::autocovmat)
  .method("vars", &kernel::variances)
  .method("sum_kernel", &kernel::sum_kernel)
  .method("multiply_kernel", &kernel::multiply_kernel)
  .method("scale_kernel", &kernel::scale_kernel)
  ;
  class_<exponential_kernel>("exp_kernel")
    .derives<kernel>("gp_kernel")
    .constructor<int, double, arma::vec, double>()
  ;
  class_<rational_quadratic_kernel>("rq_kernel")
    .derives<kernel>("gp_kernel")
    .constructor<int, double, double, double, double>()
  ;
  class_<exponential_constant_kernel>("exp_const_kernel")
    .derives<kernel>("gp_kernel")
    .constructor<int, double, double, arma::vec, double>()
  ;
  class_<sum_exponential_kernels>("sum_exp_kernels")
    .derives<kernel>("gp_kernel")
    .constructor<int, double, double, arma::vec, arma::vec, double>()
  ;
  function("sde_vi", &do_sde_inference);
}



