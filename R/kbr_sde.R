#' @importFrom KernSmooth locpoly
kbr_train <- function(x, h,
                      kernels = c("normal", "normal"),
                      bws = c(0.5,0.5), range.x = NULL) {
  if (length(kernels) == 1) {
    kernels = rep(kernels, 2)
  }
  if (length(bws) == 1) {
    bws = rep(bws, 2)
  }
  if (is.null(range.x)) {
    range.x = range(x)
  }
  useX = head(x, -1)
  indx = order(useX)
  useX = useX[indx]
  dx = diff(x)[indx]
  estimates = locpoly(useX, dx,
                      kernel = kernels[[1]], bandwidth = bws[[1]],
                      range.x = range.x)
  estimates$drift = estimates$y / h
  estimates$y = NULL

  estimates$diffusion = locpoly(useX, dx ^ 2,
                                kernel = kernels[[2]], bandwidth = bws[[2]],
                                range.x = range.x)$y
  estimates$diffusion = estimates$diffusion / h

  attr(estimates, "kernels") = kernels
  attr(estimates, "bws") = bws
  class(estimates) = "kbr_sde"
  estimates
}

mtrapz = function(x, y) {
  idx = 2:length(x)
  as.double((x[idx] - x[idx - 1]) %*% (y[idx] + y[idx - 1])) / 2
}


# Reference:
# Kernel-based regression of drift and diffusion coefficients of stochastic processes
# @param x The original time series.
# @param y A time series simulated from the estimations obtained from x.
# @param h The sampling period.
# @param drift An R function obtained through the interpolation of the drift
# estimate of the x series.
# @param kernel The kernel used in the x's parameters estimations.
# @param bw The bandwidth used in the x's parameters estimations.
#' @importFrom KernSmooth bkde
#' @importFrom KernSmooth locpoly
#' @importFrom zoo rollapply
#' @importFrom zoo zoo
calculate_delta_error = function(y, x, h, type = c("drift", "diff"),
                                 fun, kernels, bws) {
  type = match.arg(type)
  bw = switch(type, "drift" = bws[[1]],  "diff" = bws[[2]])
  tryCatch({
    yEstimates =
      kbr_train(y, h, kernels, bws)
    #useRange = c(min(x,y) - 4 * bw, max(x,y) + 4 * bw)
    useRange = c(max(min(x),min(y)),
                 min(max(x),max(y)))
    xDens = bkde(x, bandwidth = bw, range.x = useRange)
    yDens = bkde(y, bandwidth = bw, range.x = useRange)
    useSupport = yDens$x
    px = xDens$y
    px[px < 0] = 0
    py = yDens$y
    py[py < 0] = 0
    Dx = fun(useSupport)
    if (type == "drift") {
      Dy = approx(yEstimates$x, yEstimates$drift,
                  xout = useSupport, rule = 2)$y
    } else {
      Dy = approx(yEstimates$x, yEstimates$diffusion,
                  xout = useSupport, rule = 2)$y
    }
    mtrapz(useSupport, abs(Dy - Dx) * sqrt(px * py)) / mtrapz(useSupport, sqrt(px * py))
  }, error = function(e) {
    warning("NA/NaN/Inf in calculate_delta_error: returning NA")
    NA
  })
}

which_local_minima = function(x) {
  which(diff(sign(diff(x))) == 2) + 1
}

get_valid_index = function(x) {
  which(!is.na(x) & !is.nan(x) & !is.infinite(x))
}

#' @importFrom parallel makeCluster
#' @importFrom parallel stopCluster
#' @import foreach
#' @importFrom doParallel registerDoParallel
select_best_bandwidth =
  function(x, h, kernels, type = c("drift", "diff"),
           bwGrid, fixedBw, errorBw = 0.1, nSim = 500,
           solveTies = c("maxDrop", "minArg", "maxArg","minVal", "maxVal", "na"),
           nthreads,plotErrors = TRUE){

    type = match.arg(type)
    solveTies = match.arg(solveTies)

    if (nthreads > 1) {
      clu = makeCluster(nthreads)
      registerDoParallel(clu)
      `%fun%` = `%dopar%`
      on.exit({
        registerDoSEQ()
        stopCluster(clu)
      })
    } else {
      `%fun%` = `%do%`
    }

    models =
      foreach(bwIt = bwGrid, .combine = c,
              .init = list(), .packages = 'voila') %fun% {
                bwPars = switch(type,
                                "drift" = c(bwIt, fixedBw),
                                "diff" = c(fixedBw, bwIt))
                # estimate functions in an extended range to simulate new time series
                xRange = range(x) + 3 * c(-1, 1) * bwIt
                kbrEst =
                  kbr_train(x, h, kernels, bwPars, xRange)

                validIndx = intersect(get_valid_index(kbrEst$drift),
                                      get_valid_index(kbrEst$diffusion))
                kbrEst$x = kbrEst$x[validIndx]
                kbrEst$drift = kbrEst$drift[validIndx]
                kbrEst$diffusion = kbrEst$diffusion[validIndx]
                drift = approxfun(kbrEst$x, kbrEst$drift, rule = 2)
                diffusion = approxfun(kbrEst$x, kbrEst$diffusion, rule = 2)
                sims = euler_maruyama(drift, diffusion, x[[1]], h,
                                      N = length(x), nSim = nSim)
                deltaError = apply(sims, 2, calculate_delta_error,
                                   x = x, h = h, type = type,
                                   fun = ifelse(type == "drift", drift, diffusion),
                                   kernel = kernels, bws = bwPars)

                # Select only those values corresponding to the range of x
                indx = which(kbrEst$x >= min(x) & kbrEst$x <= max(x))
                kbrEst$x = kbrEst$x[indx]
                kbrEst$drift = kbrEst$drift[indx]
                kbrEst$diffusion = kbrEst$diffusion[indx]
                list(list(meanError = mean(deltaError, na.rm = TRUE),
                          model = kbrEst, bw = bwIt))
              }
    bws = sapply(models, function(x) x$bw)
    errors = sapply(models, function(x) x$meanError)
    invalidIndx = which(is.infinite(errors) |is.na(errors) | is.nan(errors))
    if (length(invalidIndx) > 0) {
      warning("There are some invalid Delta Errors (inf/na/nan)")
      allModels = models[-invalidIndx]
      bws = bws[-invalidIndx]
      errors = errors[-invalidIndx]
      if (length(allModels) == 0) {
        stop("All models have invalid Delta Errors (inf/na/nan)")
      }
    }
    deltaError = zoo::zoo(errors, bws)
    meanBwGrid = mean(diff(sort(bwGrid)))
    width = max(2, errorBw / meanBwGrid)
    if (width < length(deltaError)) {
      deltaError = zoo::rollapply(deltaError,
                                  width = width,
                                  function(x) mean(x, na.rm = TRUE))
    } else {
      warning("Not enough points for smoothing the delta-errors...Skipping")
    }
    if (all(is.na(deltaError))) {
      stop("All errors are unavailable. Could not find best model")
    }
    # best model is the one with local minima
    bestIndex = which_local_minima(deltaError)
    if (length(bestIndex) > 1) {
      bestIndex =
        switch(solveTies,
               "minVal" = bestIndex[which.min(deltaError[bestIndex])],
               "maxVal" = bestIndex[which.max(deltaError[bestIndex])],
               "minArg" = bestIndex[which.min(time(deltaError)[bestIndex])],
               "maxArg" = bestIndex[which.max(time(deltaError)[bestIndex])],
               "maxDrop" = {
                 diffError = diff(deltaError)
                 diffError = c(diffError[[1]], diffError)
                 timeDiff = time(deltaError)
                 decreaseValue = sapply(bestIndex, function(x) {
                   startDepressionIdx = max(c(1, which(diffError[1:x] > 0)),
                                            na.rm = FALSE)
                   if (is.infinite(startDepressionIdx)) {
                     Inf
                   } else {
                     as.numeric((deltaError[[x]] - deltaError[[startDepressionIdx]]) /
                                  (timeDiff[[x]] - timeDiff[[startDepressionIdx]]) )
                   }
                 })
                 bestIndex[[which.min(decreaseValue)]]
               },
               "na" = bestIndex[[1]])
    }
    if (length(bestIndex) == 0) {
      # pick global minimum
      bestIndex = which.min(deltaError)
    }
    # get the bandwidth which is closest to the best bandwidth selected
    # with deltaError
    bestModelIndex = which.min(abs(bws - time(deltaError)[[bestIndex]]))
    #for debug purposes only
    if (plotErrors) {
      plot(deltaError, main = "Delta Error Vs Bandwidth",
           xlab = "Bandwidth", ylab = "Delta Error",type="o", cex=0.2)
      points(deltaError[bestIndex], col = 2, bg = 2,
             pch = 22)
    }
    allModels = lapply(models, function(x) x$model)
    list(all = allModels,
         best = allModels[bestModelIndex],
         deltaError = deltaError)
  }



#' @export
fit_kbr_sde <- function(x, h, kernels = c("normal", "normal"),
                        driftBw = 0.5, diffBw = 0.5,
                        nSim = 500, nthreads = 1,
                        solveTiesDrift = c("maxDrop","minArg", "maxArg","minVal", "maxVal", "na"),
                        solveTiesDiff = c("maxDrop", "minArg", "maxArg","minVal", "maxVal", "na"),
                        driftErrorBw = 0.1, diffErrorBw = driftErrorBw,
                        plotErrors = TRUE) {
  if (length(driftBw) * length(diffBw) == 1) {
    return(
      kbr_train(x,h, kernels, c(driftBw, diffBw))
    )
  }

  # select best drift bw
  if (length(driftBw) > 1) {
    bw1 =  select_best_bandwidth(x = x, h = h, kernels = kernels,
                                 type = "drift",
                                 bwGrid = driftBw, fixedBw = mean(diffBw),
                                 errorBw = driftErrorBw,
                                 nSim = nSim, solveTies = solveTiesDrift,
                                 nthreads, plotErrors)
    bestDriftBw = attr(bw1$best[[1]], "bws")[[1]]
  } else {
    bestDriftBw = driftBw
    bw1 = NULL
  }
  # select best diff bw
  if (length(diffBw) > 1) {
    bw2 =  select_best_bandwidth(x = x, h = h,
                                 kernels =  kernels, type = "diff",
                                 bwGrid = diffBw, fixedBw = bestDriftBw,
                                 errorBw = diffErrorBw,
                                 nSim = nSim, solveTies = solveTiesDiff,
                                 nthreads, plotErrors)
    best = bw2$best[[1]]
  } else {
    best = bw1$best[[1]]
  }
  list(best = best, all = list(drift = bw1$all,diff = bw2$all),
       errors = list(drift = bw1$deltaError, diff = bw2$deltaError))
}


#' @export
plot.kbr_sde = function(x, which = c("drift", "diffusion"), type = "l",
                        xlab = NULL, ylab = NULL, main = NULL, ...) {
  which = match.arg(which)
  y = x[[which]]
  if (is.null(xlab)) {
    xlab = "x"
  }
  if (is.null(ylab)) {
    ylab = paste(which, "term")
  }
  if (is.null(main)) {
    main = paste("KBR estimate of the", which, "term")
  }
  plot(x$x, y,
       xlab = xlab, ylab = ylab, main = main,
       type = type, ...)
}
