// Fish population representation in the simulation.
/// @file    fishpop.cpp

#include "fishpop.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>


void fishpop::create(
  xt::xarray<int> init_ages,
  xt::xarray<int> init_counts,
  xt::xarray<double> init_weights
) {
  ages = init_ages;
  counts = init_counts;
  weights = init_weights;
}

void fishpop::set_ages(xt::xarray<int> new_ages) {
  ages = new_ages;
}

xt::xarray<int> fishpop::get_ages() {
  return ages;
}

xt::xarray<int> fishpop::get_counts() {
  return counts;
}

xt::xarray<double> fishpop::get_weights() {
  return weights;
}


