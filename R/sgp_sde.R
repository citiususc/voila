#' Create a function returning a constant value
#'
#' @param v Constant value to be returned by the function
#' @return A function returning a vector with constant value \emph{v} and
#' length equal to the input vector
#' @examples
#' fun = constant_function_factory(0)
#' # all values are TRUE
#' print(fun(1:10) == rep(0, 10))
#' @export
constant_function_factory = function(v) {
  force(v)
  function(x) {
    if (inherits(x,"matrix")) {
      rep(v, nrow(x))
    } else {
      rep(v, length(x))
    }
  }
}


#' SGP-based estimate of a Langevin Equation/SDE
#'
#' @param kernel @param kernel An object representing a gaussian process'
#' kernel, e.g. a \emph{sde_kernel} object.
#' @param xm A matrix representing the pseudo-inputs of the Sparse Gaussian
#' Process (SGP).
#' @param priorMeanFun A function returning the prior mean of the gaussian
#' process.
#' @param posteriorMean A vector representing the mean of the gaussian
#' distribution that approximates the posterior of the drift/diffusion term.
#' @param posteriorCov A matrix representing the covariance matrix of the
#' gaussian distribution that approximates the posterior of the
#' drift/diffusion term.
#' @seealso \code{\link{predict.sgp_sde}}
#' @export
sgp_sde = function(kernel, xm, priorMeanFun,
                   posteriorMean, posteriorCov) {
  UseMethod("sgp_sde", kernel)
}


#' @export
sgp_sde.default = function(kernel, xm, priorMeanFun,
                           posteriorMean, posteriorCov) {
  if (!is_valid_kernel_pointer(kernel) && !inherits(kernel, "sde_kernel")) {
    stop("A 'sde_kernel' object or a C++ kernel pointer was expected")
  }
  if (!inherits(xm, "matrix") && (length(xm) > 1)) {
    xm = matrix(xm, ncol = 1)
  }

  kmm = autocovmat(kernel, xm)

  sgp = list(xm = xm,
             kmm = kmm,
             priorMeanFun = priorMeanFun,
             kmmInv = chol2inv(chol(kmm)),
             gpKernel = kernel,
             posteriorMean = posteriorMean,
             posteriorCov = posteriorCov)
  class(sgp) = "sgp_sde"
  sgp
}



#' Predictions from drift/diffusion estimates
#'
#' @param object An \emph{sgp_sde} object representing an estimate of a
#' drift/diffusion term.
#' @param newX A matrix representing the new input values with which to predict
#' @param lognormal Logical value indicating if the \emph{sgp_sde} represents
#' a gaussian process or a lognormal process. If the \emph{sgp_sde} is a drift
#' estimate it should be set to FALSE, whereas if it is a diffusion estimate it
#' should be set to TRUE.
#' @param quantiles Vector with the quantiles that the function should return
#' to build the confidence intervals.
#' @param ... Additional parameters (currently unused).
#' @return A \emph{sde_prediction} with the mean and confidence intervals
#' of the prediction. Use the \emph{plot} function to plot the estimates.
#' @rdname predict.sgp_sde
#' @export
predict.sgp_sde = function(object, newX, lognormal = FALSE,
                           quantiles = c(0.05, 0.95), ...) {
  sparseGP = object
  if (length(quantiles) < 2) {
    stop("length of quantiles should be > 2")
  }
  gpk = sparseGP$gpKernel
  kmx = covmat(gpk, sparseGP$xm, newX)
  A = t(kmx) %*% sparseGP$kmmInv

  mu = sparseGP$priorMeanFun(newX) +
    A %*% (sparseGP$posteriorMean - sparseGP$priorMeanFun(sparseGP$xm))
  var = vars(gpk, newX) - diag(A %*% kmx) +
    diag(A %*% sparseGP$posteriorCov %*% t(A))
  qs = sapply(quantiles, FUN = function(x) qnorm(x, mu, sqrt(var)))
  # lognormal process
  if (lognormal) {
    # don't overwrite mu for the moment: it is necessary for the var calculation
    qs = exp(qs)
    mup = exp(mu + var / 2)
    var = (exp(var) - 1) * exp(2 * mu + var)
    mu = mup
  }
  gpPrediction = list(mean = mu,
                      var = var,
                      x = newX,
                      qs = qs,
                      lognormal = lognormal)
  class(gpPrediction) = "sde_prediction"
  gpPrediction
}
