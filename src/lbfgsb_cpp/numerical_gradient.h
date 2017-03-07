/*
 * Copyright Constantino Antonio Garcia 2017
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef LBFGSB_ADAPTER_NUMERICAL_GRADIENT_H
#define LBFGSB_ADAPTER_NUMERICAL_GRADIENT_H


#include <vector>

// https://en.wikipedia.org/wiki/Finite_difference_coefficient
template<class T, typename F>
T approximate_gradient(F& functor, const T& x, int accuracy = 0, double gridSpacing = 1e-6) {
    if (accuracy < 0 || accuracy > 3) {
        throw std::invalid_argument("Accuracy should be between 0 and 3");
    }
    std::vector< std::vector<double> > finiteDiffCoefs = {
            {-0.5, 0, 0.5},
            {1.0 / 12.0, -2.0 / 3.0, 0, 2.0 / 3.0, -1.0 / 12.0},
            {-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0},
            {1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0, -4.0 / 5.0, 0, 4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0}
    };
    std::vector<double> coefs = finiteDiffCoefs[accuracy];
    int nCoefs = coefs.size();

    // copy to resize to proper dimension without depending on the class of T
    T gr(x);
    // copy x since we need to modify it but it is declared constant
    T workX(x);
    int Dim = x.size();

    for (int i = 0; i < Dim; i++) {
        gr[i] = 0.0;
        for (int j = 0, k = -(accuracy + 1); j < nCoefs; j++, k++) {
            workX[i] += k * gridSpacing;
            gr[i] += coefs[j] * functor(workX);
            workX[i] = x[i];
        }
        gr[i] /= gridSpacing;
    }
    return gr;

}

#endif //LBFGSB_ADAPTER_NUMERICAL_GRADIENT_H
