#ifndef MPI_TIMETEST_CPP
#define MPI_TIMETEST_CPP

#include "mpi_timetest.h"


double time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        / 1e9;
}

double avg_time_of_mat_vec_product_mpi(size_t n, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum,
        size_t &min_mem_usage, size_t &max_mem_usage,
        size_t my_id, size_t proc_cnt) {

    std::vector<double> v, pr;
    CSR_matrix a;

    generate_vector_mpi(n, 1 + seed, v, my_id, proc_cnt);
    generate_matrix_mpi(n, 10 + seed, a, my_id, proc_cnt);

    // THIS IS NOT COMPLETE because of transfers
    min_mem_usage = max_mem_usage = 0;

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        product_mpi(n, a, v, pr, my_id, proc_cnt);
    }
    st = time() - st;

    csum = control_sum_mpi(n, pr, cs_seed, my_id, proc_cnt);

    return st / repeats;

}


double avg_time_of_linear_combination_mpi(size_t n, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum, size_t &mem_usage,
        size_t my_id, size_t proc_cnt) {

    std::vector<double> x, y;
    generate_vector_mpi(n, 1+seed, x, my_id, proc_cnt);
    generate_vector_mpi(n, 2+seed, y, my_id, proc_cnt);

    mem_usage = 3 * n * sizeof(double);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        linear_combination(0.72, 0.27, x, y); // this is cumulative sum for my laziness
        // sum of coeffs less than 1 to avoid exponential growth
    }
    st = time() - st;

    csum = control_sum_mpi(n, x, cs_seed, my_id, proc_cnt);

    return st / repeats;

}


double avg_time_of_dot_product_mpi(size_t n, size_t repeats,
        uint_fast64_t seed, double &dot, size_t &mem_usage,
        size_t my_id, size_t proc_cnt) {

    std::vector<double> x, y;
    generate_vector_mpi(n, 1+seed, x, my_id, proc_cnt);
    generate_vector_mpi(n, 2+seed, y, my_id, proc_cnt);

    mem_usage = 2 * n * sizeof(double);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        dot = dot_product_mpi(x, y);
    }
    st = time() - st;

    return st / repeats;

}

#endif
