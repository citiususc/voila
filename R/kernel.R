
# modify to add new kernels ----------------------------------------------------
kVoilaValidKernels = c("exp_kernel", "rq_kernel",
                        "sum_exp_kernels", "exp_const_kernel", "clamped_exp_lin_kernel")

is_valid_kernel_pointer = function(kernel) {
  any(sapply(paste0("Rcpp_", kVoilaValidKernels),
        FUN = inherits, x = kernel))
}

create_kernel_pointer = function(obj, ...) {
  UseMethod("create_kernel_pointer", obj)
}

kVoilaRequiredParameters = list(
  'exp_kernel' = c('amplitude','lengthScales'),
  'rq_kernel' = c('amplitude', 'alpha', 'lengthScales'),
  'sum_exp_kernels' = c('maxAmplitude','amplitude1', 'lengthScales1', 'lengthScales2'),
  'exp_const_kernel' = c('maxAmplitude', 'expAmplitude', 'lengthScales'),
  'clamped_exp_lin_kernel' = c('maxAmplitude', 'linAmplitude', 'linCenter', 'lengthScales')
)

create_kernel_pointer.default = function(type, parameters, inputDimension, epsilon,
                                         lowerBound = NULL, upperBound = NULL) {
  # correct a typical mistake
  if ('scaleLengths' %in% names(parameters)) {
    parameters$lengthScales = parameters$scaleLengths
  }
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
                             epsilon),
    'clamped_exp_lin_kernel' = new(clamped_exp_lin_kernel, inputDimension,
                                   parameters$maxAmplitude,
                                   parameters$linAmplitude, parameters$linCenter,
                                   parameters$lengthScales,
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

  # the attributes MUST appear in the order used by the cpp constructor
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
      kernelAttributes$maxAmplitude = parameters$maxAmplitude
      kernelAttributes$hyperparams = list(expAmplitude = parameters$expAmplitude,
                                          lengthScales = parameters$lengthScales)
      kernelAttributes
    },
    'clamped_exp_lin_kernel' = {
      kernelAttributes$maxAmplitude = parameters$maxAmplitude
      kernelAttributes$hyperparams = list(linAmplitude = parameters$linAmplitude,
                                          linCenter = parameters$linCenter,
                                          lengthScales = parameters$lengthScales)
      kernelAttributes
    })
}

# TODO: ADD ALSO TO THE TYPE ARGUMENT IN THE SDE_KERNEL FUNCTION
# end of required modifications for new kernels --------------------------------

#' Create a gaussian process' kernel
#'
#' Creates a kernel defining the properties of a gaussian process
#' @param type A string specifying the type of kernel to create
#' @param parameters A NAMED list with the parameters that the kernel needs for
#' its proper creation. The easiest way of seeing which parameters are required
#' is to make this function fail. For example:
#' \emph{sde_kernel("exp_kernel",list())}
#' @param inputDimension The input dimension of the kernel
#' @param epsilon A small value to be added to the diagonal of the kernel
#' covariance matrices to regularize them. This improves the numerical stability
#' of the computations.
#' @return A \emph{sde_kernel} S3 object
#' @seealso \code{\link{covmat}}, \code{\link{autocovmat}}, \code{\link{vars}},
#' \code{\link{get_hyperparams}}, \code{\link{set_hyperparams}},
#' \code{\link{decrease_upper_bound}} and \code{\link{increase_lower_bound}}
#' @export
sde_kernel = function(type = c("exp_kernel", "rq_kernel", "sum_exp_kernels",
                               "exp_const_kernel", "clamped_exp_lin_kernel"),
                      parameters, inputDimension = 1, epsilon = 0) {

  type = match.arg(type)
  if ((length(names(parameters)) < length(kVoilaRequiredParameters[[type]])) ||
      !all(names(parameters) %in% kVoilaRequiredParameters[[type]])) {
    stop(paste0("Missing parameters:'", type, "' kernel requires the NAMED parameters c('",
                paste(collapse = "', '", kVoilaRequiredParameters[[type]]), "')"))
  }
  # create kernel just to check the parameters correction (use C++ code to
  # avoid replication) and get default bounds
  junk = create_kernel_pointer(type, parameters, inputDimension, epsilon)

  kernelAttr = create_kernel_attributes(type, parameters, inputDimension, epsilon)
  kernelAttr$lowerBound = lapply(junk$get_lower_bound(), function(x) x)
  kernelAttr$upperBound = lapply(junk$get_upper_bound(), function(x) x)
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

#' Covariance matrix
#'
#' Computes the covariance matrix of the input vectors represented with
#' matrices (in which each row represents an input vector).
#'
#' @param kernel An object representing a gaussian process' kernel, e.g. a
#' \emph{sde_kernel} object.
#' @param x,y A matrix in which each row represents an input vector. Note that
#' if the inputs are univariate, a matrix with just one column should
#' be used.
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

#' Autocovariance matrix
#'
#' Computes the autocovariance matrix of the input vectors represented with
#' a matrix (in which each row represents an input vector).
#'
#' @inheritParams covmat
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


#' Variance vector
#'
#' Calculates the variances of each of the input vectors represented with
#' a matrix (in which each row represents an input vector).
#'
#' @inheritParams autocovmat
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

#' Get the hyperparameters of a kernel
#' @inheritParams autocovmat
#' @export
get_hyperparams = function(kernel) {
  UseMethod("get_hyperparams", kernel)
}

#' @export
get_hyperparams.sde_kernel = function(kernel) {
  kernel$hyperparams
}

#' Set the hyperparameters of a kernel
#' @inheritParams autocovmat
#' @param hyperparams The new hyperparameters of the kernel
#' @export
set_hyperparams = function(kernel, hyperparams) {
  UseMethod("set_hyperparams", kernel)
}

#' @export
set_hyperparams.sde_kernel = function(kernel, hyperparams) {
  if (is.list(hyperparams)) {
    hyperparams = arrange_hyperparams_list(kernel, hyperparams)
  } else {
    # hyperparams is a vector: we assume default order.
    # get name and sizes of the hyperparams
    oldHp = get_hyperparams(kernel)
    hpNames = names(oldHp)
    hpDims = sapply(oldHp, length)
    begin = c(1, head(cumsum(hpDims), -1) + 1)
    end = begin + hpDims - 1
    hyperparamsList = list()
    for (i in seq_along(hpNames)) {
     hyperparamsList[[ hpNames[i] ]] = hyperparams[ begin[i]:end[i] ]
    }
    hyperparams = hyperparamsList
  }
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
  kernel$set_hyperparams(as.numeric(hyperparams))
  kernel
}


#' Decreases the upper bound of the kernel
#'
#' Decreases the maximum values that each of the hyperparameters of the kernel
#' may take.
#'
#' @inheritParams autocovmat
#' @param upperBound A vector with the new values of the upper bound.
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

#' Decreases the lower bound of the kernel
#'
#' Decreases the minimum values that each of the hyperparameters of the kernel
#' may take.
#'
#' @inheritParams autocovmat
#' @param lowerBound A vector with the new values of the lower bound.
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

