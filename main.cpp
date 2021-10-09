#include <iostream>
#include <chrono>
#include <string>

#include "matrices.cpp"
#include "tests.cpp"


double time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        / 1e9;
}


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
}
