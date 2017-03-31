#' A simple wrapper for yuima's SDE simulation methods
#' @export
#' @importFrom yuima setModel simulate setSampling get.zoo.data
simulate_sde = function(driftExpression, diffExpression,
                        samplingPeriod, tsLength, xinit,
                        trueParameter = list()) {
  model = suppressWarnings(setModel(drift = driftExpression,
                                    diffusion = diffExpression))
  X = suppressWarnings(simulate(model,
                               xinit =  xinit,
                                sampling = setSampling(delta = samplingPeriod,
                                                       n = tsLength),
                                true.parameter = trueParameter)
  )
  as.matrix(get.zoo.data(X)[[1]], ncol = 1)
}