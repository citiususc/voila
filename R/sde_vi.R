#' Variation inference for Langevin equations/SDE
#'
#' Estimates the drift and diffusion functions of a Langevin equation from
#' a single trajectory of the process. The estimation models the drift and
#' diffusion terms as gaussian processes and makes use of Variational Inference
#' (VI) to approximate the posteriors.
#'
#' @param timeSeriesIndex Integer indicating in which of the input dimensions
#' the inference should be made.
#' @param timeSeries Matrix of size number_observations x input_dimensionality
#' representing the multivariate time series from which the estimation shall
#' be made.  Note that if the time series is univariate, a matrix with just
#' one column should be used.
#' @param samplingPeriod The sampling period of the time series
#' @param xm Matrix with the same number of columns as the \emph{timeSeries}
#' representing the initial values of the pseudo-inputs.
#' @param fKernel,sKernel An object representing a gaussian process' kernel
#' used to model the drift/diffusion term, e.g. a \emph{sde_kernel} object.
#' @param v Numeric value representing the prior mean for the diffusion
#' process.
#' @param maxIterations Integer value specifying the maximum number of iterations
#' of the method.
#' @param hyperparamIterations Integer value specifying the maximum number
#' of iterations to be used when optimizing the hyperparameters.
#' @param relTol Relative tolerance used to assess the convergence of the
#' method.
#' @return A list with two \emph{sgp_sde} objects representing the drift
#' and diffusion estimates (\emph{drift} and \emph{diff} fields) and other
#' useful information related with the inference process.
#' @export
sde_vi = function(timeSeriesIndex = 1, timeSeries, samplingPeriod,
                  xm, fKernel, sKernel, v,
                  maxIterations = 20, hyperparamIterations = 5,
                  relTol = 1e-6) {
  if (!inherits(timeSeries, "matrix")) {
    timeSeries = matrix(timeSeries, ncol = 1)
    timeSeriesIndex = 1
  }
  if (!inherits(xm, "matrix")) {
    xm = matrix(xm, ncol = 1)
  }

  check_sde_vi_params(timeSeriesIndex, timeSeries, samplingPeriod,
                      xm, fKernel, sKernel, v)

  if (inherits(fKernel, "sde_kernel")) {
    fKernelPointer = create_kernel_pointer.sde_kernel(fKernel)
  } else {
    fKernelPointer = fKernel
  }
  if (inherits(sKernel, "sde_kernel")) {
    sKernelPointer = create_kernel_pointer.sde_kernel(sKernel)
  } else {
    sKernelPointer = sKernel
  }
  result = do_sde_inference(timeSeriesIndex - 1, timeSeries, samplingPeriod,
                            xm, fKernelPointer, sKernelPointer, v,
                            maxIterations, hyperparamIterations, relTol)

  fKernel = set_hyperparams(fKernel, result$fHp)
  sKernel = set_hyperparams(sKernel, result$sHp)

  driftSgp = sgp_sde(fKernel, result$xm, constant_function_factory(0),
                     result$fMean, result$fCov)
  diffSgp = sgp_sde(sKernel, result$xm, constant_function_factory(result$v),
                    result$sMean, result$sCov)

  list(drift = driftSgp, diff = diffSgp,
       likelihoodLowerBound = result$Ls,
       driftKernelParameters = list(mean = result$fMean, cov = result$fCov),
       diffKernelParameters = list(mean = result$sMean, cov = result$sCov,
                                   priorMean = result$v),
       inducingPoints = result$xm,
       driftHyperparams = result$fHp,
       diffHyperparams = result$sHp)
}

check_sde_vi_params = function(timeSeriesIndex, timeSeries, samplingPeriod,
                               xm, fKernel, sKernel, v) {
  #
  if (ncol(timeSeries) != ncol(xm)) {
    stop("The number of columns of 'timeSeries' and 'xm' does not match")
  }
  if (timeSeriesIndex < 0 || timeSeriesIndex > ncol(timeSeries)) {
    stop("'timeSeriesIndex' out of bounds")
  }
  if (samplingPeriod <= 0) {
    stop("Invalid 'samplingPeriod' ('samplingPeriod' <= 0)")
  }
  if (!is_valid_kernel_pointer(fKernel) && !inherits(fKernel, "sde_kernel")) {
    stop("A 'sde_kernel' object or a C++ kernel pointer was expected")
  }
  if (!is_valid_kernel_pointer(sKernel) && !inherits(sKernel, "sde_kernel")) {
    stop("A 'sde_kernel' object or a C++ kernel pointer was expected")
  }
}