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

#' @export
predict.voila_preproc = function(object, input, inverse = FALSE) {
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




