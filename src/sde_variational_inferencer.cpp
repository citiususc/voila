#define ARMA_DONT_PRINT_ERRORS
#include "sde_variational_inferencer.h"
#include "utils.h"

sde_variational_inferencer::sde_variational_inferencer(const arma::mat& timeSeries,
                                                       double samplingPeriod)
  : problem(),
    mX(timeSeries),
    mHeadX(timeSeries.submat(0, 0, timeSeries.n_rows - 2, timeSeries.n_cols - 1)),
    mDX(arma::diff(timeSeries, 1)),
    mSamplingPeriod(samplingPeriod),
    mDXSize(timeSeries.n_rows - 1),
    mFKernel(timeSeries.n_cols), mSKernel(timeSeries.n_cols) {

}

// TODO: change
void sde_variational_inferencer::set_max_iterations(int maxIterations) {
  mMaxIt = maxIterations;
}

void sde_variational_inferencer::set_verbose_level(int verboseLevel) {
  mVerboseHp = verboseLevel;
}

Rcpp::List sde_variational_inferencer::do_inference(int targetIndex, arma::mat& xm,
                                              kernel& fKernel, kernel& sKernel,
                                              double v){
 Rcpp::Rcout << "Let's do some VI" << std::endl;
  // TODO: sanitize inputs
  mTarget = mDX.col(targetIndex);
  mFKernel = fKernel;
  mSKernel = sKernel;

  mFKernel.greet();
  mNoPseudoInputs = xm.n_rows;
  mNoFHyperparameters = mFKernel.get_hyperparams().n_elem;
  mNoSHyperparameters = mSKernel.get_hyperparams().n_elem;
  // set the variables related to the class problem properly
  // TODO: create a protected set_input_dimension that resets also
  // the bounds
  mInputDimension = mNoFHyperparameters + mNoSHyperparameters + mNoPseudoInputs + 1;
  mLowerBound = arma::vec(mInputDimension);
  mUpperBound = arma::vec(mInputDimension);
  double limitValue = std::numeric_limits<double>::max();

    set_lower_bound(
    arma::join_cols(arma::join_cols(mFKernel.get_lower_bound(),
                                    mSKernel.get_lower_bound()),
                    arma::join_cols(rep_value(mHeadX.min(), mNoPseudoInputs),
                                    arma::vec({-limitValue}))
    ));
  Rcpp::Rcout << mLowerBound << std::endl;
  set_upper_bound(
    arma::join_cols(arma::join_cols(mFKernel.get_upper_bound(),
                                    mSKernel.get_upper_bound()),
                    arma::join_cols(rep_value(mHeadX.max(), mNoPseudoInputs),
                                    arma::vec({limitValue}))
    ));

  Rcpp::Rcout << mUpperBound << std::endl;

  calculate_model_matrices(xm);

  mPosteriorFMean.resize(mNoPseudoInputs);
  mPosteriorFMean.fill(0);
  mPosteriorFCov = mKmm;
  mPosteriorSCov = mJmm;
  mPosteriorSMean.resize(mNoPseudoInputs);
  mPosteriorSMean.fill(v);
  mLikelihoodLowerBounds.reserve(2 * mMaxIt);
  mLikelihoodLowerBounds.push_back(get_lower_bound(v));
  if (mVerbose) {
     std::cout << "Initial Lower Bound L = " << mLikelihoodLowerBounds[0] << std::endl;
  }
  int nIt = 1;
  double L, previousL = mLikelihoodLowerBounds[0];

  mSolver.set_verbose_level(mVerboseHp);
  mSolver.set_projected_gradient_tolerance(0);

  for (; nIt <= mMaxIt; nIt++) {
    update_distributions(v);
    mLikelihoodLowerBounds.push_back( get_lower_bound(v) );
    if (mVerbose) {
      distributions_update_message(nIt);
    }
    // optimize hyperparams
    arma::vec hp = arma::join_cols(
      arma::join_cols(mFKernel.get_hyperparams(), mSKernel.get_hyperparams()),
      arma::join_cols(matrix_to_vector(xm), arma::vec({v}))
    );

    Rcpp::Rcout << "R before optim: " << std::endl;
    for (int iii = 0; iii < hp.n_elem; iii++) {
      Rcpp::Rcout << hp[iii] << " ";
    }
    Rcpp::Rcout << std::endl;
    // Rcpp::Rcout << hp << std::endl;

    // mSolver.set_verbose_level(1000);
    // TODO: CHANGE VARIABLES
    mSolver.set_max_iterations(5);
    mSolver.optimize(*this, hp);
    mSolver.set_max_iterations(500);
    //mSolver.set_verbose_level(-1);


    Rcpp::Rcout << "R after optim: " << std::endl;
    for (int iii = 0; iii < hp.n_elem; iii++) {
      Rcpp::Rcout << hp[iii] << " ";
    }
    Rcpp::Rcout << std::endl;
    //mSolver.set_verbose_level(-1);
    // ensure that all parameters are up to date
    mFKernel.set_hyperparams(hp.subvec(0, mNoFHyperparameters - 1));
    mSKernel.set_hyperparams(hp.subvec(mNoFHyperparameters,
                                      mNoFHyperparameters + mNoSHyperparameters - 1));
    arma::vec xmVector = hp.subvec(mNoFHyperparameters + mNoSHyperparameters, hp.n_elem - 2);
    xm = vector_to_matrix(xmVector, mNoPseudoInputs, mHeadX.n_cols);
    v = hp(hp.n_elem - 1);

    calculate_model_matrices(xm);

    mLikelihoodLowerBounds.push_back( get_lower_bound(v) );
    if (mVerbose) {
      hyperparameters_optimization_message(nIt);
    }
    if (std::fabs((mLikelihoodLowerBounds.back() - previousL) / previousL) < mRelativeTolerance) {
      if (mVerbose) {
        Rcpp::Rcout << "Convergence" << std::endl;
      }
      break;
    }
    previousL = mLikelihoodLowerBounds.back();
  }

  return Rcpp::List::create(Rcpp::Named("Ls") = mLikelihoodLowerBounds,
                            Rcpp::Named("fMean") = mPosteriorFMean,
                            Rcpp::Named("sMean") = mPosteriorSMean,
                            Rcpp::Named("fCov") = mPosteriorFCov,
                            Rcpp::Named("sCov") = mPosteriorSCov,
                            Rcpp::Named("xm") = xm,
                            Rcpp::Named("fHp") = mFKernel.get_hyperparams(),
                            Rcpp::Named("sHp") = mSKernel.get_hyperparams(),
                            Rcpp::Named("v") = v
  );
}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v) const {
  return calculate_E_vector(v, mPosteriorSMean, mPosteriorSCov, mB, mHii);
}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v,
                                                            const arma::vec& sMean,
                                                            const arma::mat& sCov,
                                                            const arma::mat& B,
                                                            const arma::vec& Hii) {
 // Rcpp::Rcout << "E" << std::endl;
  int noPseudoInputs = sMean.n_elem;
  int dxSize = Hii.n_elem;
  arma::colvec sMeanMinusV(noPseudoInputs);
  std::transform(sMean.begin(), sMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  arma::colvec BgMean = B * sMeanMinusV;
  arma::mat gCovByB = sCov * B.t();
  arma::colvec Evector(dxSize);
  for (int i = 0; i < dxSize; i++) {
    double bb = 0;
    for (int j = 0; j < noPseudoInputs; j++) {
      bb += B(i,j) * gCovByB(j,i);
    }
    Evector(i) = std::exp(-v - BgMean(i) + 0.5 * (bb + Hii(i)));
  }
  return Evector;
}

arma::colvec sde_variational_inferencer::calculate_ksi_vector() const {
  return calculate_ksi_vector(mTarget, mSamplingPeriod, mPosteriorFMean,
                              mPosteriorFCov, mA, mQii);
}

arma::colvec sde_variational_inferencer::calculate_ksi_vector(
    const arma::vec& dxTarget, double samplingPeriod, const arma::vec& fMean,
    const arma::mat& fCov, const arma::mat& A, const arma::vec& Qii) {
  int dxSize = Qii.n_elem;
  arma::colvec Af = A * fMean;
  arma::mat Fmat = fCov + fMean * fMean.t();
  double h2 = samplingPeriod * samplingPeriod;

  arma::colvec result(dxSize);
  for (int i = 0; i < dxSize; i++) {
    arma::rowvec ai = A.row(i);
    result(i) = std::pow(dxTarget(i), 2) - 2 * samplingPeriod * dxTarget(i) * Af(i) +
      h2 * (arma::dot(ai, Fmat * ai.t()) +  Qii(i));
  }
  return result;
}

double sde_variational_inferencer::get_lower_bound(double v) const {
  return get_lower_bound(mTarget, mSamplingPeriod,
                         mPosteriorFMean, mPosteriorFCov, mA, mQii, mKmmInv, v,
                         mPosteriorSMean, mPosteriorSCov, mB, mHii, mJmmInv);
}

double sde_variational_inferencer::get_lower_bound(const arma::vec& dxTarget,
                                                   double samplingPeriod,
                                                   const arma::vec& fMean,
                                                   const arma::mat& fCov,
                                                   const arma::mat& A,
                                                   const arma::vec& Qii,
                                                   const arma::mat& kmmInv,
                                                   double v,
                                                   const arma::vec& sMean,
                                                   const arma::mat& sCov,
                                                   const arma::mat& B,
                                                   const arma::vec& Hii,
                                                   const arma::mat& jmmInv) {
  arma::vec E = calculate_E_vector(v, sMean, sCov, B, Hii);
  arma::vec ksi = calculate_ksi_vector(dxTarget, samplingPeriod, fMean, fCov, A, Qii);

  int noPseudoInputs = fMean.n_elem;
  arma::colvec sMeanMinusV(noPseudoInputs);
  std::transform(sMean.begin(), sMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  int dxSize = dxTarget.n_elem;
  double log2pi = log(2 * arma::datum::pi );
  double logEntConstant = noPseudoInputs * log(2 * arma::datum::pi * arma::datum::e);

  return  0.5 * (-arma::dot(E, ksi) / samplingPeriod
                 -dxSize * v - sum(B * sMeanMinusV) +
                 -dxSize * log(samplingPeriod) - dxSize * log2pi +
                 -arma::trace(jmmInv * sCov) +
                 -arma::dot(sMeanMinusV, jmmInv * sMeanMinusV) +
                 -noPseudoInputs * log2pi  + log(arma::det(jmmInv))
                 -arma::trace(kmmInv * fCov) +
                 -arma::dot(fMean, kmmInv * fMean) +
                 -noPseudoInputs * log2pi  + log(arma::det(kmmInv)) +
                 log(det(fCov)) + log(det(sCov)) + 2 * logEntConstant);
}

double sde_variational_inferencer::operator()(const arma::vec& x) {
  // Rcpp::Rcout << "*** hyperparams = ";
  // for (int ii = 0; ii < mUpperBound.n_elem; ii ++ ) {
  //   Rcpp::Rcout << x[ii] << " ";
  // }
  // Rcpp::Rcout << std::endl;
//   Rcpp::Rcout << "*** lb = ";
//   for (int ii = 0; ii < mUpperBound.n_elem; ii ++ ) {
//     Rcpp::Rcout << mLowerBound[ii] << " ";
//   }
//   Rcpp::Rcout << std::endl;
//   Rcpp::Rcout << "*** ub = ";
//   for (int ii = 0; ii < mUpperBound.n_elem; ii ++ ) {
//     Rcpp::Rcout << mUpperBound[ii] << " ";
//   }
  // Rcpp::Rcout << std::endl;
  mFKernel.set_hyperparams(x.subvec(0, mNoFHyperparameters - 1));
  mSKernel.set_hyperparams(x.subvec(mNoFHyperparameters,
                                    mNoFHyperparameters + mNoSHyperparameters - 1));
  arma::vec xmVector = x.subvec(mNoFHyperparameters + mNoSHyperparameters, x.n_elem - 2);
  arma::mat xm = vector_to_matrix(xmVector, mNoPseudoInputs, mHeadX.n_cols);
  double v = x(x.n_elem - 1);

  try {
    calculate_kernel_matrices(mHeadX, mFKernel, xm,
                              mKmm, mKmmInv, mKnm, mA, mQii);
    //Rcpp::Rcout << "F is fine" << std::endl;
    calculate_kernel_matrices(mHeadX, mSKernel,
                              xm, mJmm, mJmmInv, mJnm, mB, mHii);
    //Rcpp::Rcout << "S is fine" << std::endl;
    // note the minus sign due to the fact that mSolver minimizes and we
    // want to maximize
  } catch (const std::exception& e) {
    // there is some singular matrix due to a bad setting of hyperparams.
    // return worst value, which is a large number since mSolver minimizes
    return std::numeric_limits<float>::max();
  }
  //Rcpp::Rcout << "Returning L" << std::endl;
  return -get_lower_bound(
      mTarget, mSamplingPeriod,
      mPosteriorFMean, mPosteriorFCov,
      mA, mQii, mKmmInv, v,
      mPosteriorSMean, mPosteriorSCov,
      mB, mHii, mJmmInv
  );

}

void sde_variational_inferencer::calculate_model_matrices(const arma::mat& xm) {
  calculate_kernel_matrices(mHeadX, mFKernel, xm, mKmm, mKmmInv, mKnm, mA, mQii);
  calculate_kernel_matrices(mHeadX, mSKernel, xm, mJmm, mJmmInv, mJnm, mB, mHii);
}


void sde_variational_inferencer::calculate_kernel_matrices(const arma::mat& headX,
                                                           const kernel& ker,
                                                           const arma::mat& xm,
                                                           arma::mat& kmm,
                                                           arma::mat& kmmInv,
                                                           arma::mat& knm,
                                                           arma::mat& A,
                                                           arma::vec& Qii) {
  //std::cout << "l = " <<  ker.get_hyperparams() << std::endl;
  //std::cout << xm << std::endl;
  kmm = ker.autocovmat(xm);
  //std::cout << kmm << std::cout;
  //std::cout << kmm << std::endl;
  kmmInv = arma::inv_sympd(kmm);
  // stabilize computations
  kmmInv = (kmmInv + kmmInv.t()) / 2.0;
  knm = ker.covmat(headX, xm);
  A = knm * kmmInv;

  int dxSize = headX.n_rows;
  arma::colvec auxVec = arma::zeros<arma::colvec>(dxSize);
  for (int i = 0; i < dxSize; i++) {
    for (int j = 0; j < xm.n_rows; j++) {
      for (int k = 0; k < xm.n_rows; k++) {
        auxVec(i) += knm(i,j) * kmmInv(j,k) * knm(i,k);
      }
    }
  }
  Qii = ker.variances(headX) - auxVec;
  Qii(find(Qii < 0)).zeros();
}

void sde_variational_inferencer::update_distributions(double v) {
  // Update distribution of the drift
  arma::vec E = calculate_E_vector(v);
  arma::mat posteriorFCovInv = mKmmInv + mSamplingPeriod * mA.t() * diagmat(E) * mA;
  // Rcpp::Rcout << posteriorFCovInv << std::endl;
  posteriorFCovInv = ( posteriorFCovInv + posteriorFCovInv.t() ) / 2.0;
  mPosteriorFCov = arma::inv_sympd(posteriorFCovInv);
  mPosteriorFCov = (mPosteriorFCov + mPosteriorFCov.t()) / 2.0;
  mPosteriorFMean = ( ((E % mTarget).t() * mA) * mPosteriorFCov).t();

  // Update distribution of the diffusion
  arma::vec ksi = calculate_ksi_vector();
  arma::vec logksi = arma::log(ksi);
  std::transform(logksi.begin(), logksi.end(),
                 logksi.begin(),
                 bind2nd(std::plus<double>(), -v));

  // use laplace approximation to find the distribution of the diffusion
  arma::vec cvector = arma::exp(logksi + mHii / 2.0);

  laplace_objective_function lof(mPosteriorSMean.n_elem,
                                 cvector,  v,
                                 *this);
  mSolver.optimize(lof, mPosteriorSMean);

  arma::mat Bp = mB, Bpp = mB;
  arma::vec meanMinusV(mPosteriorSMean.n_elem);
  std::transform(mPosteriorSMean.begin(), mPosteriorSMean.end(),
                 meanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));
  arma::vec auxVec = arma::exp(-mB * meanMinusV);
  for (int i = 0; i < mB.n_cols; i++) {
    Bp.col(i) = mB.col(i) % cvector;
    Bpp.col(i) = mB.col(i) % auxVec;
  }
  arma::mat auxMat = 1.0 / (2.0 * mSamplingPeriod) * (Bp.t() * Bpp) + mJmmInv;
  auxMat = (auxMat + auxMat.t()) / 2.0;
  mPosteriorSCov = arma::inv_sympd(auxMat);
  mPosteriorSCov = (mPosteriorSCov + mPosteriorSCov.t()) / 2.0;

  // Rcpp::Rcout << "Fm = " << mPosteriorFMean << std::endl;
  // Rcpp::Rcout << "Sm= " << mPosteriorSMean << std::endl;
  // Rcpp::Rcout << "Fcov = " << mPosteriorFCov << std::endl;
  // Rcpp::Rcout << "Scov = " << mPosteriorSCov << std::endl;
  // Rcpp::Rcout << "********************" << std::endl << std::endl;

}


void sde_variational_inferencer::distributions_update_message(int nIt) {
  Rcpp::Rcout << std::setprecision(8) <<
    "Iteration " << nIt << "| Distributions update | L = " <<
    mLikelihoodLowerBounds[2 * (nIt - 1) + 1] << std::endl;
}

void sde_variational_inferencer::hyperparameters_optimization_message(int nIt) {
   Rcpp::Rcout << std::setprecision(8) <<
     "Iteration " << nIt << "| Hyperparameter optimization | L = " <<
      mLikelihoodLowerBounds[2 * nIt] << std::endl;
}