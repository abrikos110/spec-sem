#ifndef MPI_TIMETEST_H
#define MPI_TIMETEST_H

#include <chrono>
#include "mpi_operations.h"


double time();

double avg_time_of_mat_vec_product_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, size_t repeats, uint_fast64_t seed,
    uint_fast64_t cs_seed, std::vector<double> &csum, size_t &min_mem_usage,
    size_t &max_mem_usage, size_t my_id, size_t proc_cnt);

double avg_time_of_linear_combination_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, size_t repeats, uint_fast64_t seed,
    uint_fast64_t cs_seed, std::vector<double> &csum, size_t &mem_usage,
    size_t my_id, size_t proc_cnt);

double avg_time_of_dot_product_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, size_t repeats, uint_fast64_t seed,
    double &dot, size_t &mem_usage, size_t my_id, size_t proc_cnt);

#endif
