#' @method plot sde_prediction
#' @export
plot.sde_prediction = function(x, includeConf = TRUE, col = 1,
                               type = "l", ylim = NULL, ...) {
  plot_sde_prediction('plot', x, includeConf, col, type, ylim, ...)
}


#' @export
lines.sde_prediction = function(x, includeConf = TRUE, col = 1,
                               type = "l", ylim = NULL, ...) {
  plot_sde_prediction('lines', x, includeConf, col, type, ylim, ...)
}


plot_sde_prediction = function(funName, x, includeConf, col, type, ylim, ...) {
    FUN = match.fun(funName)
    indx = order(x$x)
    support = x$x[indx]
    xinf = x$qs[indx,which.min(x$qs[1,])]
    xsup = x$qs[indx,which.max(x$qs[1,])]
    if (is.null(ylim)) {
      if (includeConf) {
        ylim = c(min(xinf,na.rm = TRUE), max(xsup, na.rm = TRUE))
      } else {
        ylim = range(x$mean)
      }
    }
    FUN(support, x$mean[indx], ylim = ylim, type = type, ...)
    if (includeConf) {
      if (funName == "plot") {
        polygon(c(support, rev(support)),
                c(xinf, rev(xsup)),
                col = "gray90")
      } else {
        dotList = list(...)
        # pick a different lty ...
        validLty =  c('blank', 'solid', 'dashed', 'dotted', 'dotdash', 'longdash', 'twodash')
        if ('lty' %in% names(dotList)) {
          # it is enought to do the which(...) since
          # the numeric values start at 0. Hence, which will return the next integer
          # with respect to the numeric value of the lty
          if (is.numeric(dotList$lty)) {
            lty = dotList$lty + 1
          } else {
            lty = which(dotList$lty == validLty)
          }
        } else {
          lty = 2
        }
        lines(support, xinf, lty = lty, col = col)
        lines(support, xsup, lty = lty, col = col)
      }
    }
    lines(support, x$mean[indx], col = col, type = type, ...)
}


