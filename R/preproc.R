#' Preprocess a time series
#'
#' Center and scale a time series
#'
#' @param x time series to preprocess
#' @param method Vector with the preprocessing methods to apply.
#' @param useCenter,useScale Numeric value specifying the center/scale to use
#' for the preprocessing instead of learning these values from the time
#' series \emph{x}
#' @return \emph{preproc} returns a  \emph{voila_preproc} object that can
#' be used to transform new data using \emph{predict}.
#' @examples
#' plot(do_events, main = "DO events")
#' pp = preproc(do_events)
#' scaled_do_events = predict(pp, do_events)
#' plot(scaled_do_events, main = "Scaled DO events")
#' # recover the original DO events
#' original_do_events = predict(pp, scaled_do_events, inverse = TRUE)
#' @export
preproc = function(x, method = c("center", "scale"),
                   useCenter = NULL, useScale = NULL) {
  validMethods = c("center", "scale")
  if (!all(method %in% validMethods)) {
    stop("invalid method")
  }
  object = list(sd = ifelse("scale" %in% method,
                            ifelse(is.null(useScale), sd(x), useScale),
                            1),
                mean = ifelse("center" %in% method,
                              ifelse(is.null(useCenter),mean(x), useCenter),
                              0)
  )
  class(object) = "voila_preproc"
  object
}


#' @rdname preproc
#' @param object A \emph{voila_preproc} object
#' @param input The input to be transformed using \emph{object}
#' @param inverse Logical value: Is the data being trasformed back?
#' @param ... Additional parameters (currently ignored)
#' @export
predict.voila_preproc = function(object, input, inverse = FALSE, ...) {
  if (inherits(input, "sde_prediction")) {
    input$x = predict(object, input$x, inverse = inverse)
    scalingFactor = ifelse(input$lognormal, object$sd ^ 2, object$sd)
    scalingFactor = ifelse(inverse, scalingFactor, 1 / scalingFactor)
    input$mean = input$mean * scalingFactor
    input$var = input$var * scalingFactor ^ 2
    input$qs = input$qs * scalingFactor
    return(input)
  }
  # default case
  if (inverse) {
    input * object$sd + object$mean
  } else {
    (input - object$mean) / object$sd
  }
}




