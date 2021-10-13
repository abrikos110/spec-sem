#include <iostream>
#include <string>

#include "matrices.cpp"
#include "tests.cpp"
#include "timetest.cpp"


int main(int argc, char **args) {
    int N = 0;
    uint_fast64_t seed = 22, cs_seed = 999;

    std::cout.precision(16);

    if (argc < 2) {
        N = 10;
    }
    else {
        N = std::stoi(args[1]);
        if (argc >= 3) {
            seed = std::stoi(args[2]);
            if (argc >= 4) {
                cs_seed = std::stoi(args[3]);
            }
        }
    }

    std::cout << "Usage: " << args[0] << " N [seed] [control_sum_seed]\n\n";
    std::cout << "N : " << N << "\n\n";

    test_dot_product(N, seed, N <= 10);
    test_mat_vec_product(N, seed, cs_seed, N <= 10);
    test_linear_combination(N, seed, cs_seed, N <= 10);

    std::vector<double> csum;
    int rep = 100;
    int n = 1000000;
    std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_mat_vec_product(n, rep, seed, cs_seed, csum) << std::endl;
    PRINT_VECTOR(csum, "control sum");

    n = 10000000;
    std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_linear_combination(n, rep, seed, cs_seed, csum) << std::endl;
    PRINT_VECTOR(csum, "control sum");

    double dot = -1;
    std::cout << "avg time of dot product, n=" << n << ", repeats=" << rep
        << ": " << avg_time_of_dot_product(n, rep, seed, dot) << std::endl
        << "dot : " << dot << std::endl;
}
