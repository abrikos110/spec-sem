#ifndef MATRICES_CPP
#define MATRICES_CPP

#include <vector>


struct CSR_matrix {
    std::vector<size_t> IA, JA;
    std::vector<double> values;

    CSR_matrix(size_t n = 0)
        : IA(n, 0) {}

    size_t size() const {
        return IA.size();
    }

    void set_size(size_t n) {
        IA.resize(n, 0);
    }

    size_t JA_begin(size_t i) const {
        return IA[i];
    }
    size_t JA_end(size_t i) const {
        return i < size() - 1 ? IA[i+1] : JA.size();
    }

    double getv(size_t i, size_t j) const {
        for (size_t k = JA_begin(i); k < JA_end(i); ++k) {
            if (JA[k] == j) {
                return values[k];
            }
        }
        return 0;
    }
};


void product(
        const CSR_matrix &A,
        const std::vector<double> &v,
        std::vector<double> &ans) {

    ans.resize(A.size(), 0);
    #pragma omp parallel for
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
    #pragma omp parallel for reduction(+:ans)
    for (size_t i = 0; i < a.size(); ++i) {
        ans += a[i] * b[i];
    }
    return ans;
}


void linear_combination(double a, double b,
        std::vector<double> &x, const std::vector<double> &y) {
    // x = a * x + b * y
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = a * x[i] + b * y[i];
    }
}

#endif
