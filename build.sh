#!/usr/bin/env sh

mkdir build 2>/dev/null
mpic++ main.cpp csr_matrix.cpp mpi_operations.cpp mpi_timetest.cpp mpi_test.cpp -fopenmp -std=c++17 -Wall -Wextra -Wpedantic -O3 -Ofast -o build/mpi_main

echo
echo Compiled to build/mpi_main
