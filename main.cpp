#include <iostream>
#include <string>
#include <omp.h>

#include "matrices.cpp"
#include "tests.cpp"
#include "timetest.cpp"

#ifndef CS_SEED
#define CS_SEED 999
#endif

#ifndef SEED
#define SEED 22
#endif


int main() {
#ifdef NUM_THREADS
    omp_set_num_threads(NUM_THREADS);
#else
    #warning "NUM_THREADS is not defined"
#endif

    std::cout.precision(16);

#ifdef TEST
#define N 9
    std::cout << "N : " << N << "\n\n";

    test_dot_product(N, SEED, N <= 10);
    test_mat_vec_product(N, SEED, CS_SEED, N <= 10);
    test_linear_combination(N, SEED, CS_SEED, N <= 10);
#else
    std::vector<double> csum;
    int rep = 100;
    int n = 1000;//60000000;
    std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_mat_vec_product(n, rep, SEED, CS_SEED, csum) << std::endl;
    PRINT_VECTOR(csum, "  control sum");

    std::cout << "avg time of linear combination, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_linear_combination(n, rep, SEED, CS_SEED, csum) << std::endl;
    PRINT_VECTOR(csum, "  control sum");

    double dot = -1;
    std::cout << "avg time of dot product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_dot_product(n, rep, SEED, dot) << std::endl
        << "  dot : " << dot << std::endl;
#endif
}
