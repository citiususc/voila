
#' Polynomial based SDEs Estimation
#'
#' @param timeSeries A univariate vector representing the time series
#' @param samplingPeriod The sampling period of the time series
#' @param timeTs A vector with the occurrence time of each sample of the time
#' series. Can be used as an alternative to \emph{samplingPeriod}.
#' @param nDrift,nDiff Maximum order of the polynomial used to fit the
#' drift/diffusion term.
#' @param direction The mode of stepwise search, can be one of "both",
#' "backward", or "forward". See \code{\link[stats]{step}}.
#' @return A list with two \emph{poly_sde} objects representing the estimates
#' for the drift and the diffusion terms. The \emph{poly_sde} objects can be
#' used with \emph{predict} to get numerical estimates of the drift/diffusion
#' functions.
#' @export
fit_polynomial_sde <- function(timeSeries, samplingPeriod = NULL,
                               timeTs = NULL,
                               nDrift = 1, nDiff = 1,
                               direction = "both"){
  # only for one dimensional data
  if (inherits(timeSeries, "matrix")) {
    if (ncol(timeSeries) > 1) {
      stop("Poly method is only for one dimensional data")
    } else {
      timeSeries = as.numeric(timeSeries)
    }
  }
  if (is.null(samplingPeriod) && is.null(timeTs)) {
    stop("samplingPeriod or timeTs should be specified")

  }
  N = length(timeSeries)
  dtimeSeries = diff(timeSeries)
  x = head(timeSeries, -1)
  if (!is.null(samplingPeriod)) {
    timeTs = seq(0,length.out = N, by = samplingPeriod)
    dtime = rep(samplingPeriod, length.out = N - 1)
  } else {
    if (length(timeTs) != length(timeSeries)) {
      stop("incorrect length of timeTs")
    }
    dtime = diff(timeTs)
  }
  if (nDrift > 0) {
    driftPoly = poly(x, nDrift)
    data = cbind.data.frame(dtimeSeries / dtime, driftPoly)
    colnames(data) = c("y", paste0("c",1:nDrift))
    driftFit = lm(y ~ ., data = data)
    # predictors selection
    driftFit = step(driftFit, direction = direction, trace = 0)
    # fit only with selected predictors
    driftFit = lm(formula(driftFit), data)
  } else {
    data = data.frame('y' = dtimeSeries / dtime)
    driftFit = lm(y ~ 1, data = data)
    driftPoly = NULL
  }
  e = (diff(timeSeries) - driftFit$fitted.values * dtime) ^ 2 / dtime
  if (nDiff > 0) {
    diffPoly = poly(x, nDiff)
    data = cbind.data.frame(e, diffPoly)
    colnames(data) = c("y", paste0("c", 1:nDiff))
    diffFit = lm(y ~ ., data = data)
    # predictors selection
    diffFit = step(diffFit, direction = direction, trace = 0)
    # fit only with selected predictors
    diffFit = lm(formula(diffFit), data)
  } else {
    data = data.frame('y' = e)
    diffFit = lm(y ~ 1, data = data)
    diffPoly = NULL
  }
  driftResult = list(fit = driftFit, polyModel = driftPoly, polyOrder = nDrift)
  class(driftResult) = "poly_sde"
  attr(driftResult, "type") = "drift"
  diffResult = list(fit = diffFit, polyModel = diffPoly, polyOrder = nDiff)
  class(diffResult) = "poly_sde"
  attr(diffResult, "type") = "diffusion"
  list(drift = driftResult, diff = diffResult)
}


#' @export
predict.poly_sde = function(object, newX, ...) {
  if (!is.null(object$polyModel)) {
    newX = as.data.frame(predict(object$polyModel, newX))
    colnames(newX) = paste0("c", 1:object$polyOrder)
  } else {
    newX = as.data.frame(newX)
  }
  predict(object$fit, newdata = newX)
}
