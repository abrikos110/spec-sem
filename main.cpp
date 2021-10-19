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

#ifndef N
#define N 11
#endif


int main() {
#ifdef NUM_THREADS
    omp_set_num_threads(NUM_THREADS);
#else
    #warning "NUM_THREADS is not defined"
#endif

    //int N = 0;
    //uint_fast64_t SEED = 22, CS_SEED = 999;

    std::cout.precision(16);

    /* if (argc < 2) {
        N = 11;
    }
    else {
        N = std::stoi(args[1]);
        if (argc >= 3) {
            SEED = std::stoi(args[2]);
            if (argc >= 4) {
                CS_SEED = std::stoi(args[3]);
            }
        }
    } */

    /* std::cout << "N : " << N << "\n\n";

    test_dot_product(N, SEED, N <= 10);
    test_mat_vec_product(N, SEED, CS_SEED, N <= 10);
    test_linear_combination(N, SEED, CS_SEED, N <= 10); */

    std::vector<double> csum;
    int rep = 100;
    int n = 10000;
    std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_mat_vec_product(n, rep, SEED, CS_SEED, csum) << std::endl;
    PRINT_VECTOR(csum, "  control sum");

    std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_linear_combination(n, rep, SEED, CS_SEED, csum) << std::endl;
    PRINT_VECTOR(csum, "  control sum");

    double dot = -1;
    std::cout << "avg time of dot product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_dot_product(n, rep, SEED, dot) << std::endl
        << "  dot : " << dot << std::endl;
}
