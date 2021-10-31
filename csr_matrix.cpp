#ifndef CSR_MATRIX_CPP
#define CSR_MATRIX_CPP

#include "csr_matrix.h"


double CSR_matrix::getv(size_t i, size_t j) const {
    for (size_t k = JA_begin(i); k < JA_end(i); ++k) {
        if (JA[k] == j) {
            return values[k];
        }
    }
    return 0;
}

#endif
