#include "sde_variational_inferencer.h"

sde_variational_inferencer::sde_variational_inferencer(const arma::mat& timeSeries,
                                                       double samplingPeriod)
  : mX(timeSeries),
    mHeadX(timeSeries.submat(0, 0, timeSeries.n_rows - 2, timeSeries.n_cols - 1)),
    mDX(arma::diff(timeSeries, 1)),
    mSamplingPeriod(samplingPeriod),
    mDXSize(timeSeries.n_rows - 1) {

}


void sde_variational_inferencer::do_inference(int targetIndex, arma::mat& xm,
                                              kernel& fKernel, kernel& sKernel,
                                              double v){
  Rcpp::Rcout << "Let's do some VI" << std::endl;
  // TODO: sanitize inputs
  mNoPseudoInputs = xm.n_rows;
  mNoFHyperparameters = fKernel.get_hyperparams().n_elem;
  mNoSHyperparameters = sKernel.get_hyperparams().n_elem;

  calculate_model_matrices(fKernel, sKernel, xm);

  mPosteriorFMean.resize(mNoPseudoInputs);
  mPosteriorFMean.fill(0);
  mPosteriorFCov = mKmm;
  mPosteriorSCov = mJmm;
  mPosteriorSMean.resize(mNoPseudoInputs);
  mPosteriorSMean.fill(v);

  Rcpp::Rcout << get_lower_bound(fKernel, sKernel, v) << std::endl;

}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v) const {
  Rcpp::Rcout << "E" << std::endl;
  arma::colvec sMeanMinusV(mNoPseudoInputs);
  std::transform(mPosteriorSMean.begin(), mPosteriorSMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  arma::colvec BgMean = mB * sMeanMinusV;
  arma::mat gCovByB = mPosteriorSCov * mB.t();
  arma::colvec Evector(mDXSize);
  for (int i = 0; i < mDXSize; i++) {
    double bb = 0;
    for (int j = 0; j < mNoPseudoInputs; j++) {
      bb += mB(i,j) * gCovByB(j,i);
    }
    Evector(i) = std::exp(-v -BgMean(i) + 0.5 * (bb + mHii(i)));
  }

  return Evector;
}

arma::colvec sde_variational_inferencer::calculate_ksi_vector() const {
  Rcpp::Rcout << "ksi" << std::endl;
  arma::colvec Af = mA * mPosteriorFMean;
  Rcpp::Rcout << "ksi" << std::endl;
  arma::mat Fmat = mPosteriorFCov + mPosteriorFMean * mPosteriorFMean.t();
  double h2 = mSamplingPeriod * mSamplingPeriod;

  arma::colvec result(mDXSize);
  for (int i = 0; i < mDXSize; i++) {
    arma::rowvec ai = mA.row(i);
    result(i) = std::pow(mDX(i), 2) - 2 * mSamplingPeriod * mDX(i) * Af(i) +
      h2 * (arma::dot(ai, Fmat * ai.t()) +  mQii(i));
  }
  return result;
}

double sde_variational_inferencer::get_lower_bound(const kernel& fKernel,
                                                   const kernel& sKernel,
                                                   const double v) const {
  Rcpp::Rcout << "get_lower_bound" << std::endl;
  arma::colvec E = calculate_E_vector(v);
  arma::colvec ksi = calculate_ksi_vector();
  arma::colvec sMeanMinusV(mNoPseudoInputs);
  std::transform(mPosteriorSMean.begin(), mPosteriorSMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  double log2pi = log(2 * arma::datum::pi );
  double logEntConstant = mNoPseudoInputs *
    log(2 * arma::datum::pi * arma::datum::e);
  return  0.5 * (-arma::dot(E, ksi) / mSamplingPeriod
                 -mDXSize * v - sum(mB * sMeanMinusV) +
                 -mDXSize * log(mSamplingPeriod) - mDXSize * log2pi +
                 -arma::trace(mJmmInv * mPosteriorSCov) +
                 -arma::dot(sMeanMinusV, mJmmInv * sMeanMinusV) +
                 -mNoPseudoInputs * log2pi  + log(arma::det(mJmmInv))
                 -arma::trace(mKmmInv * mPosteriorFCov) +
                 -arma::dot(mPosteriorFMean, mKmmInv * mPosteriorFMean) +
                 -mNoPseudoInputs * log2pi  + log(arma::det(mKmmInv)) +
                  log(det(mPosteriorFCov)) + log(det(mPosteriorSCov)) +
                  2 * logEntConstant);
}

void sde_variational_inferencer::calculate_model_matrices(const kernel& fKer,
                                                          const kernel& sKer,
                                                          const arma::mat& xm) {
  calculate_kernel_matrices(fKer, xm, mKmm, mKmmInv, mKnm, mA, mQii);
  calculate_kernel_matrices(sKer, xm, mJmm, mJmmInv, mJnm, mB, mHii);
}


void sde_variational_inferencer::calculate_kernel_matrices(const kernel& ker,
                                                           const arma::mat& xm,
                                                           arma::mat& kmm,
                                                           arma::mat& kmmInv,
                                                           arma::mat& knm,
                                                           arma::mat& A,
                                                           arma::vec& Qii) {
  kmm = ker.autocovmat(xm);
  kmmInv = arma::inv_sympd(kmm);
  // stabilize computations
  kmmInv = (kmmInv + kmmInv.t()) / 2.0;
  knm = ker.covmat(mHeadX, xm);
  A = knm * kmmInv;

  arma::colvec auxVec = arma::zeros<arma::colvec>(mDXSize);
  for (int i = 0; i < mDXSize; i++) {
    for (int j = 0; j < xm.n_rows; j++) {
      for (int k = 0; k < xm.n_rows; k++) {
        auxVec(i) += mKnm(i,j) * mKmmInv(j,k) * mKnm(i,k);
      }
    }
  }
  Qii = ker.variances(mHeadX) - auxVec;
  Qii(find(Qii < 0)).zeros();
}
