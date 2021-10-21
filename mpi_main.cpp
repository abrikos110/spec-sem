#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_operations.cpp"


int main(int argc, char **args) {
    int process_cnt, my_id;
    handle_res(MPI_Init(&argc, &args));
    handle_res(MPI_Comm_size(MPI_COMM_WORLD, &process_cnt));
    handle_res(MPI_Comm_rank(MPI_COMM_WORLD, &my_id));

    char processor_name[256];
    int processor_name_len;
    handle_res(MPI_Get_processor_name(processor_name, &processor_name_len));

    if (!my_id) {
        std::cout << "Number of processes: " << process_cnt
            << "\nMy id: " << my_id
            << "\nProcessor name: " << processor_name << std::endl;
    }

    std::cout.precision(16);

    size_t n = 1000000;
    uint_fast64_t seed = 22, cs_seed = 999;
    test_dot_product_mpi(n, seed, my_id, process_cnt, n <= 10);
    test_linear_combination_mpi(n, seed, cs_seed, my_id, process_cnt, n <= 10);
    test_mat_vec_product_mpi(n, seed, cs_seed, my_id, process_cnt, n <= 10);

    MPI_Finalize();
}
