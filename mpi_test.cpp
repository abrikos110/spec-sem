#ifndef MPI_TEST_CPP
#define MPI_TEST_CPP

#include "mpi_test.h"


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

void test_dot_product_mpi(size_t n, size_t nx, size_t ny, size_t px, size_t py,
    uint_fast64_t seed, size_t my_id, size_t proc_cnt, bool debug) {

    if (!my_id) {
        std::cout << "MPI Dot product test\n";
    }
    comm_data cd(n, my_id, proc_cnt);
    init_l2g_part(cd, nx, ny, px, py);

    std::vector<double> v, h;
    generate_vector(cd, v, 1+seed);
    generate_vector(cd, h, 2+seed);

    if (debug) {
        sleep_ms(100 * my_id + 100);
        std::cout << "my id: " << my_id << "\n";
        PRINT_VECTOR(v, "    v");
        PRINT_VECTOR(h, "    h");
    }

    double dot = dot_product(h, v);
    if (!my_id) {
        std::cout << "dot(h, v) = " << dot << "\n\n";
    }
}


void test_linear_combination_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, uint_fast64_t seed, uint_fast64_t cs_seed,
    size_t my_id, size_t proc_cnt, bool debug) {

    comm_data cd(n, my_id, proc_cnt);
    init_l2g_part(cd, nx, ny, px, py);

    std::vector<double> v, h;
    generate_vector(cd, v, 1+seed);
    generate_vector(cd, h, 2+seed);

    if (!my_id) {
        std::cout << "MPI Linear combination test\n";
    }
    if (debug) {
        sleep_ms(100 * my_id + 100);
        std::cout << "my id: " << my_id << "\n";
        PRINT_VECTOR(h, "    h");
        PRINT_VECTOR(v, "    v");
    }
    linear_combination(3.14, 2.78, h, v);
    if (debug) {
        PRINT_VECTOR(h, "h = 3.14*h + 2.78*v");
    }
    auto cs = control_sum_mpi(cd, h, cs_seed);
    if (!my_id) {
        PRINT_VECTOR(cs, "control sum of h");
        std::cout << std::endl;
    }
}


void test_mat_vec_product_mpi(size_t n, size_t nx, size_t ny,
    size_t px, size_t py, uint_fast64_t seed, uint_fast64_t cs_seed,
    size_t my_id, size_t proc_cnt, bool debug) {

    std::vector<double> v, pr;
    CSR_matrix b;
    comm_data cd(n, my_id, proc_cnt);

    init(cd, nx, ny, px, py, seed, b, v);
    if (!my_id) {
        std::cout << "MPI Matrix-vector product test\n";
    }
    update(cd, v);

    pr.resize(b.size(), 0);
    product(cd, b, v, pr);

    if (debug) {
        sleep_ms(100 * my_id + 100);
        std::cout << "my id: " << my_id << "\n";

        PRINT_VECTOR(cd.part, "  cd.part");
        PRINT_VECTOR(b.IA, "  IA");
        PRINT_VECTOR(b.JA, "  JA");
        PRINT_VECTOR(b.values, "  A ");

        std::cout << "[\n";
        for (size_t i = 0; i < cd.n_own; ++i) {
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
    auto cs = control_sum_mpi(cd, pr, cs_seed);
    if (!my_id) {
        PRINT_VECTOR(cs, "control sum of A v");
        std::cout << std::endl;
    }
}

#endif
