#ifndef MATRICES_CPP
#define MATRICES_CPP

#include "matrices.h"


double CSR_matrix::getv(size_t i, size_t j) const {
    for (size_t k = JA_begin(i); k < JA_end(i); ++k) {
        if (JA[k] == j) {
            return values[k];
        }
    }
    return 0;
}


void product(
        const CSR_matrix &A,
        const std::vector<double> &v,
        std::vector<double> &ans) {

    ans.resize(A.size(), 0);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t k = A.JA_begin(i); k < A.JA_end(i); ++k) {
            ans[i] += A.values[k] * v[A.JA[k]];
        }
    }
}


double dot_product(
        const std::vector<double> &a,
        const std::vector<double> &b) {

    double ans = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:ans)
#endif
    for (size_t i = 0; i < a.size(); ++i) {
        ans += a[i] * b[i];
    }
    return ans;
}


void linear_combination(double a, double b,
        std::vector<double> &x, const std::vector<double> &y) {
    // x = a * x + b * y
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = a * x[i] + b * y[i];
    }
}

#endif
