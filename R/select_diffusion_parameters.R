#' helps with the selection of the hyperparameters for the diffusion covariance matrix and
#' @export
select_diffusion_parameters = function(x, samplingPeriod,
                                       priorOnSd,
                                       responseVariableIndex = 1,
                                       varX = NULL) {
  if (is.matrix(x)) {
    if (is.null(varX)) {
      varX = mad(diff(as.numeric(x[,responseVariableIndex]))) ^ 2
    }
  } else {
    x = as.numeric(x)
    if (is.null(varX)) {
      varX = mad(diff(x)) ^ 2
    }
  }

  kernelAmplitude = as.numeric( log(1 + (priorOnSd * samplingPeriod / varX) ^ 2) )
  list(v = log(varX / samplingPeriod) - (kernelAmplitude / 2),
       kernelAmplitude =  kernelAmplitude
  )
}



