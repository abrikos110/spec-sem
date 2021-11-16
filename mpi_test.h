#ifndef MPI_TEST_H
#define MPI_TEST_H

#include "mpi_operations.h"
#include "my_time.h"

void test_dot_product_mpi(size_t n, size_t nx, size_t ny, size_t px, size_t py,
    uint_fast64_t seed, size_t my_id, size_t proc_cnt, bool debug);

void test_linear_combination_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, uint_fast64_t seed, uint_fast64_t cs_seed,
    size_t my_id, size_t proc_cnt, bool debug);

void test_mat_vec_product_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, uint_fast64_t seed, uint_fast64_t cs_seed,
    size_t my_id, size_t proc_cnt, bool debug);

#endif
