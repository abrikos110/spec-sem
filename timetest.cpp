#ifndef TIMETEST_CPP
#define TIMETEST_CPP

#include <vector>
#include <chrono>

#include "matrices.cpp"
#include "tests.cpp"


double time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        / 1e9;
}


double avg_time_of_mat_vec_product(size_t N, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum) {

    std::vector<double> v, pr;
    CSR_matrix a;

    generate_vector(N, seed, v);
    generate_matrix(N, seed, a);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        product(a, v, pr);
    }
    st = time() - st;

    csum = control_sum(pr, cs_seed);

    return st / repeats;
}


double avg_time_of_linear_combination(size_t N, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum) {

    // control sums are different with different repeats
    std::vector<double> x, y;
    generate_vector(N, seed, x);
    generate_vector(N, seed, y);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        linear_combination(0.72, 0.27, x, y); // this is cumulative sum for my laziness
        // sum of coeffs less than 1 to avoid exponential growth
    }
    st = time() - st;

    csum = control_sum(x, cs_seed);

    return st / repeats;
}


double avg_time_of_dot_product(size_t N, size_t repeats,
        uint_fast64_t seed, double &dot) {

    std::vector<double> x, y;
    generate_vector(N, seed, x);
    generate_vector(N, seed, y);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        dot = dot_product(x, y);
    }
    st = time() - st;

    return st / repeats;
}

#endif
