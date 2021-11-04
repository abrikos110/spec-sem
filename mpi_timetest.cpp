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

    size_t n_own;
    std::vector<size_t> l2g, g2l, part;
    std::vector<std::pair<size_t, size_t> > ask;
    init(n, n_own, a, l2g, g2l, part, ask, my_id, proc_cnt);
    update(n_own, v, l2g, g2l, part, ask, my_id, proc_cnt);

    l2g.clear();
    part.clear();
    ask.clear();

    // calculate overall sizes of arrays
    size_t sz, ja_sz = 0, g2l_sz = 0, ia_sz = 0, v_sz = 0;
    sz = a.JA.size();
    handle_res(MPI_Allreduce(&sz, &ja_sz, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));
    sz = g2l.size();
    handle_res(MPI_Allreduce(&sz, &g2l_sz, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));
    sz = a.IA.size();
    handle_res(MPI_Allreduce(&sz, &ia_sz, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));
    sz = v.size();
    handle_res(MPI_Allreduce(&sz, &v_sz, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));

    max_mem_usage = ja_sz * (sizeof(double) + sizeof(size_t) + sizeof(double))
        + ia_sz * sizeof(double);
    min_mem_usage = ja_sz * sizeof(double) + g2l_sz * sizeof(size_t)
        + sizeof(double) * v_sz + ia_sz * sizeof(double);

    double st = time();
    pr.resize(a.size(), 0);
    for (size_t i = 0; i < repeats; ++i) {
        product_mpi(a, v, pr, g2l);
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
        linear_combination(0.72, 0.27, x, y);
        // this is cumulative sum for my laziness
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
