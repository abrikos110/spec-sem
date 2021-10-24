#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_operations.h"


int main(int argc, char **args) {
    int process_cnt, my_id;
    handle_res(MPI_Init(&argc, &args));
    handle_res(MPI_Comm_size(MPI_COMM_WORLD, &process_cnt));
    handle_res(MPI_Comm_rank(MPI_COMM_WORLD, &my_id));


    int n = 1000, /* rep = 100, num_threads = 8,*/ seed = 1, cs_seed = 2;
    bool debug = false;
    if (argc == 1) {
        if (!my_id) {
            std::cout << "Usage: " << args[0]
                << " n [--seed=SEED]"
                << " [--cs_seed=CS_SEED] [--debug]\n\n"
                << "  --debug  print matrices and vectors to stdout\n"
                << "  --seed=SEED  set seed for generating data\n"
                << "  --cs_seed=CS_SEED  set seed for control sums\n";

            std::cout << std::endl;
        }

        goto main_exit;
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg(args[i]);
        if (arg == "--debug") {
            debug = true;
        }
        else if (arg.find("--cs_seed=") == 0) {
            cs_seed = std::stoi(arg.substr(10));
        }
        else if (arg.find("--seed=") == 0) {
            seed = std::stoi(arg.substr(7));
        }
    }

    if (argc > 1) {
        n = std::stoi(args[1]);
    }


    char processor_name[256];
    int processor_name_len;
    handle_res(MPI_Get_processor_name(processor_name, &processor_name_len));

    if (!my_id) {
        std::cout << "Number of processes: " << process_cnt
            << "\nMy id: " << my_id
            << "\nProcessor name: " << processor_name << std::endl;
    }

    std::cout.precision(16);

    test_dot_product_mpi(n, seed, my_id, process_cnt, debug);
    test_mat_vec_product_mpi(n, seed, cs_seed, my_id, process_cnt, debug);
    test_linear_combination_mpi(n, seed, cs_seed, my_id, process_cnt, debug);

main_exit:
    MPI_Finalize();
}
