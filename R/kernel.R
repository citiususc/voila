
# modify to add new kernels ----------------------------------------------------
kSgpsdeValidKernels = c("exp_kernel", "rq_kernel",
                        "sum_exp_kernels", "exp_const_kernel")

is_valid_kernel_pointer = function(kernel) {
  any(sapply(paste0("Rcpp_", kSgpsdeValidKernels),
        FUN = inherits, x = kernel))
}

create_kernel_pointer = function(obj, ...) {
  UseMethod("create_kernel_pointer", obj)
}

kSgpsdeRequiredParameters = list(
  'exp_kernel' = c('amplitude','lengthScales'),
  'rq_kernel' = c('amplitude', 'alpha', 'lengthScales'),
  'sum_exp_kernels' = c('maxAmplitude','amplitude1', 'lengthScales1', 'lengthScales2'),
  'exp_const_kernel' = c('maxAmplitude', 'expAmplitude', 'lengthScales')
)

create_kernel_pointer.default = function(type, parameters, inputDimension, epsilon,
                                         lowerBound = NULL, upperBound = NULL) {
  kernel = switch(
    type,
    'exp_kernel' = new(exp_kernel, inputDimension, parameters$amplitude,
                       parameters$lengthScales, epsilon),
    'rq_kernel' =  new(rq_kernel, inputDimension, parameters$amplitude,
                       parameters$alpha, parameters$lengthScales, epsilon),
    'sum_exp_kernels' = new(sum_exp_kernels, inputDimension, parameters$maxAmplitude,
                            parameters$amplitude1, parameters$lengthScales1,
                            parameters$lengthScales2, epsilon),
    'exp_const_kernel' = new(exp_const_kernel, inputDimension, parameters$maxAmplitude,
                             parameters$expAmplitude, parameters$lengthScales,
                             epsilon)
  )
  if (!is.null(lowerBound)) {
    kernel$increase_lower_bound(as.numeric(lowerBound))
  }
  if (!is.null(upperBound)) {
    kernel$decrease_upper_bound(as.numeric(upperBound))
  }
  kernel
}

create_kernel_attributes = function(type, parameters, inputDimension, epsilon) {
  # common to all kernels
  kernelAttributes = list(inputDimension = inputDimension, epsilon = epsilon)

  switch(
    type,
    'exp_kernel' =  {
      kernelAttributes$amplitude = parameters$amplitude
      kernelAttributes$hyperparams = list(lengthScales = parameters$lengthScales)
      kernelAttributes
    },
    'rq_kernel' = {
      kernelAttributes$amplitude = parameters$amplitude
      kernelAttributes$hyperparams = list(alpha = parameters$alpha,
                                          lengthScales = parameters$lengthScales)
      kernelAttributes
    },
    'sum_exp_kernels' = {
      kernelAttributes$maxAmplitude = parameters$maxAmplitude
      kernelAttributes$hyperparams = list(amplitude1 =  parameters$amplitude1,
                                          lengthScales1 = parameters$lengthScales1,
                                          lengthScales2 = parameters$lengthScales2)
      kernelAttributes
    },
    'exp_const_kernel' = {
      kernelAttributes$amplitude = parameters$maxAmplitude
      kernelAttributes$hyperparams = list(expAmplitude = parameters$expAmplitude,
                                          lengthScales = parameters$lengthScales)
      kernelAttributes
    })
}

# TODO: ADD ALSO TO THE TYPE ARGUMENT IN THE SDE_KERNEL FUNCTION

# end of required modifications for new kernels --------------------------------

#' @export
sde_kernel = function(type = c("exp_kernel", "rq_kernel",
                               "sum_exp_kernels", "exp_const_kernel"),
                      parameters, inputDimension = 1, epsilon = 0) {

  type = match.arg(type)
  if ((length(names(parameters)) < length(kSgpsdeRequiredParameters[[type]])) ||
      !all(names(parameters) %in% kSgpsdeRequiredParameters[[type]])) {
    stop(paste0("Missing parameters:'", type, "' kernel requires the NAMED parameters c('",
                paste(collapse = "', '", kSgpsdeRequiredParameters[[type]]), "')"))
  }
  # create kernel just to check the parameters correction (use C++ code to
  # avoid replication) and get default bounds
  junk = create_kernel_pointer(type, parameters, inputDimension, epsilon)

  kernelAttr = create_kernel_attributes(type, parameters, inputDimension, epsilon)
  kernelAttr$lowerBound = list(as.numeric(junk$get_lower_bound()))
  kernelAttr$upperBound = list(as.numeric(junk$get_upper_bound()))
  names(kernelAttr$upperBound) = names(kernelAttr$lowerBound) = names(kernelAttr$hyperparams)

  attr(kernelAttr, 'type') = type
  class(kernelAttr) = 'sde_kernel'
  kernelAttr
}


create_kernel_pointer.sde_kernel = function(kernel) {
  parameters = kernel
  parameters = c(parameters, kernel$hyperparams)

  create_kernel_pointer.default(attr(kernel, 'type'), parameters,
                                kernel$inputDimension, kernel$epsilon,
                                kernel$lowerBound, kernel$upperBound)
}


#' @export
autocovmat = function(kernel, x) {
  UseMethod("autocovmat", kernel)
}

#' @export
autocovmat.sde_kernel = function(kernel, x) {
  kernelPointer = create_kernel_pointer(kernel)
  kernelPointer$autocovmat(x)
}

#' @export
autocovmat.default = function(kernel, x) {
  # use default function to join in a single entry all the kernel pointers
  if (!is_valid_kernel_pointer(kernel)) {
    stop("A C++ kernel pointer was expected")
  }
  kernel$autocovmat(x)
}

#' @export
covmat = function(kernel, x, y) {
  UseMethod("covmat", kernel)
}


#' @export
covmat.sde_kernel = function(kernel, x, y) {
  kernelPointer = create_kernel_pointer(kernel)
  kernelPointer$covmat(x, y)
}

#' @export
covmat.default = function(kernel, x, y) {
  # use default function to join in a single entry all the kernel pointers
  if (!is_valid_kernel_pointer(kernel)) {
    stop("A C++ kernel pointer was expected")
  }
  kernel$covmat(x, y)
}


#' @export
vars = function(kernel, x) {
  UseMethod("vars", kernel)
}

#' @export
vars.sde_kernel = function(kernel, x) {
  kernelPointer = create_kernel_pointer(kernel)
  kernelPointer$vars(x)
}

#' @export
vars.default = function(kernel, x) {
  # use default function to join in a single entry all the kernel pointers
  if (!is_valid_kernel_pointer(kernel)) {
    stop("A C++ kernel pointer was expected")
  }
  kernel$vars(x)
}


#' @export
get_hyperparams = function(kernel, x) {
  UseMethod("set_hyperparams", kernel)
}

#' @export
get_hyperparams.sde_kernel = function(kernel, x) {
  kernel$hyperparams
}


#' @export
set_hyperparams = function(kernel, x) {
  UseMethod("set_hyperparams", kernel)
}

#' @export
set_hyperparams.sde_kernel = function(kernel, hyperparams) {
  hyperparams = arrange_hyperparams_list(kernel, hyperparams)
  # avoid replication of code using the hyperparameters' checks included in
  # C++ code
  kernelPointer = create_kernel_pointer(kernel)
  kernelPointer$set_hyperparams(as.numeric(hyperparams))
  # if no exception is given the parameters are correct...
  kernel$hyperparams = hyperparams
  kernel
}

#' @export
set_hyperparams.default = function(kernel, hyperparams) {
  # use default function to join in a single entry all the kernel pointers
  if (!is_valid_kernel_pointer(kernel)) {
    stop("A C++ kernel pointer was expected")
  }
  kernel$set_hyperparams(hyperparams)
  kernel
}


#' @export
decrease_upper_bound = function(kernel, upperBound) {
  UseMethod("decrease_upper_bound", kernel)
}

#' @export
decrease_upper_bound.sde_kernel = function(kernel, upperBound) {
  upperBound = arrange_hyperparams_list(kernel, upperBound)
  # avoid replication of code using the checks included in C++ code
  kernelPointer = create_kernel_pointer(kernel)
  kernelPointer$decrease_upper_bound(as.numeric(upperBound))
  # if no exception is given the parameters are correct...
  kernel$upperBound = upperBound
  kernel
}

#' @export
increase_lower_bound = function(kernel, lowerBound) {
  UseMethod("increase_lower_bound", kernel)
}

#' @export
increase_lower_bound = function(kernel, lowerBound) {
  lowerBound = arrange_hyperparams_list(kernel, lowerBound)
  # avoid replication of code using the checks included in C++ code
  kernelPointer = create_kernel_pointer(kernel)
  kernelPointer$increase_lower_bound(as.numeric(lowerBound))
  # if no exception is given the parameters are correct...
  kernel$lowerBound = lowerBound
  kernel
}


arrange_hyperparams_list = function(kernel, hyperparams) {
    if ((length(names(hyperparams)) < length(kernel$hyperparams)) ||
      !all(names(hyperparams) %in% names(kernel$hyperparams))) {
    stop(paste0("Missing hyperparameter in  'hyperparams'. ",
                "The following NAMED hyperparameters must be present: c('",
                paste0(collapse = "', '",  names(kernel$hyperparams))),
         "')")
  }
  hyperparams[names(kernel$hyperparams)]
}




