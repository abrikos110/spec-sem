#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_operations.cpp"


#define handle_res(res) handle_res_f(res, #res " FAILED")
void handle_res_f(int res, const char *err_msg) {
    if (res != MPI_SUCCESS) {
        std::cerr << err_msg << " " << res << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
}


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

    int n = 100, seed = 22, cs_seed = 999;
    test_dot_product_mpi(n, seed, my_id, process_cnt, n <= 10);
    test_linear_combination_mpi(n, seed, cs_seed, my_id, process_cnt, n <= 10);

    MPI_Finalize();
}
