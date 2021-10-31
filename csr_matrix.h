#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

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

    double getv(size_t i, size_t j) const;
};

#endif
