# useful for creating functions that return a constant value
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

#' @export
predict.sgp_sde = function(sparseGP, newX, lognormal = FALSE,
                           quantiles = c(0.05, 0.95)) {
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
