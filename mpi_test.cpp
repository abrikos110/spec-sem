#ifndef MPI_TEST_CPP
#define MPI_TEST_CPP

#include "mpi_test.h"


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

void test_dot_product_mpi(size_t n, uint_fast64_t seed,
        size_t my_id, size_t proc_cnt, bool debug) {

    if (!my_id) {
        std::cout << "MPI Dot product test\n";
    }
    std::vector<double> v, h;
    generate_vector_mpi(n, 1+seed, v, my_id, proc_cnt);
    generate_vector_mpi(n, 2+seed, h, my_id, proc_cnt);

    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }

    double dot = dot_product_mpi(h, v);
    if (!my_id) {
        std::cout << "dot(h, v) = " << dot << "\n\n";
    }
}


void test_linear_combination_mpi(
        size_t n, uint_fast64_t seed,
        uint_fast64_t cs_seed,
        size_t my_id, size_t proc_cnt, bool debug) {

    std::vector<double> v, h;
    generate_vector_mpi(n, 1+seed, v, my_id, proc_cnt);
    generate_vector_mpi(n, 2+seed, h, my_id, proc_cnt);

    if (!my_id) {
        std::cout << "MPI Linear combination test\n";
    }
    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }
    linear_combination(3.14, 2.78, h, v);
    if (debug) {
        PRINT_VECTOR(h, "h = 3.14*h + 2.78*v");
    }
    auto cs = control_sum_mpi(n, h, cs_seed, my_id, proc_cnt);
    if (!my_id) {
        PRINT_VECTOR(cs, "control sum of h");
        std::cout << std::endl;
    }
}


void test_mat_vec_product_mpi(
        size_t n, uint_fast64_t seed, uint_fast64_t cs_seed,
        size_t my_id, size_t proc_cnt, bool debug) {

    std::vector<double> v, pr;
    CSR_matrix b(n);

    if (!my_id) {
        std::cout << "MPI Matrix-vector product test\n";
    }

    generate_matrix_mpi(n, 10 + seed, b, my_id, proc_cnt);
    generate_vector_mpi(n, 1+seed, v, my_id, proc_cnt);


    size_t n_own;
    std::vector<size_t> l2g, g2l, part;
    std::vector<std::pair<size_t, size_t> > ask;

    init(n, n_own, b, l2g, g2l, part, ask, my_id, proc_cnt);
    update(n_own, v, l2g, g2l, part, ask, my_id, proc_cnt);

    l2g.clear();
    part.clear();
    ask.clear();

    pr.resize(b.size(), 0);
    product_mpi(b, v, pr, g2l);

    if (debug) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200 * my_id));
        PRINT_VECTOR(b.IA, "IA");
        PRINT_VECTOR(b.JA, "JA");
        PRINT_VECTOR(b.values, "A ");

        std::cout << "[\n";
        for (size_t i = 0; i < n; ++i) {
            std::cout << " [";
            for (size_t j = 0; j < n; ++j) {
                std::cout << b.getv(i, j) << ",  ";
            }
            std::cout << "],\n";
        }
        std::cout << "]\n";

        PRINT_VECTOR(v, "v  ");
        PRINT_VECTOR(pr, "A v");
    }
    auto cs = control_sum_mpi(n, pr, cs_seed, my_id, proc_cnt);
    if (!my_id) {
        PRINT_VECTOR(cs, "control sum of A v");
        std::cout << std::endl;
    }
}

#endif
