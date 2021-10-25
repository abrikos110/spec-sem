#!/usr/bin/env sh

mkdir build 2>/dev/null
mpic++ mpi_main.cpp matrices.cpp mpi_operations.cpp -std=c++17 -Wall -Wextra -Wpedantic -O3 -Ofast -o build/mpi_main

echo
echo Compiled to build/mpi_main
