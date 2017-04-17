#' Parameter selection for the diffusion term
#'
#' The selection of the amplitude parameters for gaussian process modelling the
#' diffusion term is complicated since we have to link the covariance matrix of
#' a log-normal random variable with our prior belief about the \emph{variance}
#' of the diffusion term (here, variance may be interpreted as a reasonable
#' estimate for the squared-amplitude of the diffusion). This function helps
#' with the selection of those parameters.
#'
#' @param x A time series that may be modelled with a Langevin equation.
#' @param samplingPeriod the sampling period of the time series
#' @param priorOnSd An estimate for the maximum change in amplitude of the
#' diffusion term.
#' @param responseVariableIndex Integer indicating in which of the input
#' dimensions the inference will be made.
#' @param varX The selection of the hyperparametes is based on the variance
#' of the differentiated time series. Although it is estimated by the procedure,
#' the user may specify a different value using this parameter.
#' @return A list with a mean (\emph{v}) and a covariance amplitude for the
#' log-gaussian process (\emph{kernelAmplitude}).
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



