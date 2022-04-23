
// Main for the anglersim fish population simulator.
/// @file    main.cpp

#include <mpi.h>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

// Runs like:
// mpiexec -n <number of processes> ./anglersim 

int main(int argc, char* argv[]) {
  int world_size, my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Print out information about MPI_COMM_WORLD
  std::cout << "World Size: " << world_size << "   Rank: " << my_rank << std::endl;

  xt::xarray<double> arr1
    {
      {1.0, 2.0, 3.0},
      {2.0, 5.0, 7.0},
      {2.0, 5.0, 7.0}
    };

  xt::xarray<double> arr2
    {5.0, 6.0, 7.0};
  xt::xarray<double> arr3
    {5.0, 6.0, 7.0};

  xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  res = xt::linalg::dot(arr2, arr3);

  std::cout << res << std::endl;

  MPI_Finalize();

  return 0;
}
