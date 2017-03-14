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