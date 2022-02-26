
// Main for the anglersim fish population simulator.
/// @file    main.cpp
/// @author  Robert J. Hardwick

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

int main() {
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

  return 0;
}
