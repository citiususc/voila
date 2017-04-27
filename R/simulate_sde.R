#' Simulate Langevin Equations
#'
#' A simple wrapper for yuima's SDE simulation methods
#' @param driftExpression,diffExpression A string/\emph{expression} specifying
#' the drift and diffusion terms
#' @param samplingPeriod The sampling period of the resulting time series
#' @param tsLength The length of the resulting time series
#' @param stateVariable Vector of names of the variables used in the drift and
#' diffusion equations.
#' @param xinit Initial value for the resulting time series
#' @param trueParameter A list containing the true parameters of the drift and
#' diffusion expressions (in the case that there are some unknown values).
#' @return A matrix containing the simulated trajectory.
#' @seealso \code{\link[yuima]{simulate}}
#' @export
#' @examples
#' simTs = simulate_sde("-x", "sqrt(2)", 0.001, 1000)
#' plot.ts(simTs, xlab = "Time t", ylab = "x(t)",
#'         main = "Ornstein-Uhlenbeck process")
#' @importFrom yuima setModel simulate setSampling get.zoo.data
simulate_sde = function(driftExpression, diffExpression,
                        samplingPeriod, tsLength,
                        stateVariable = 'x',
                        xinit,
                        trueParameter = list()) {
  model = suppressWarnings(setModel(drift = driftExpression,
                                    diffusion = diffExpression,
                                    state.variable = stateVariable))
  if (missing(xinit) && length(stateVariable) > 1) {
    xinit = rnorm(length(stateVariable))
  }
  X = suppressWarnings(simulate(model,
                                xinit = xinit,
                                sampling = setSampling(delta = samplingPeriod,
                                                       n = tsLength),
                                true.parameter = trueParameter)
  )
  # transform the list into a matrix
  sapply(get.zoo.data(X), function(x) x)
}

# f and g should be vectorized!!
# used in kbr_sde
euler_maruyama = function(f, g, x1, h, N, nSim) {
  noise = matrix(rnorm(N * nSim, 0, sd = sqrt(h)), ncol = nSim)
  Xsim = matrix(0, ncol = nSim, nrow = N)
  if (length(x1) == 1) {
    Xsim[1,] = rep(x1, nSim)
  } else if (length(x1) != nSim) {
    stop("length(x1) != nSim")
  } else {
    Xsim[1,] = x1
  }
  for (i in 2:N) {
    Xsim[i,] = Xsim[i - 1, ] + h * f(Xsim[i - 1,]) + noise[i - 1,] * g(Xsim[i - 1,])
  }
  Xsim
}
