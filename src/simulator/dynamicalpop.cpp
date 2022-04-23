// The dynamical population model driving the simulation.
/// @file    dynamicalpop.cpp

#include "fishpop.h"
#include "dynamicalpop.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>


void dynamicalpop::increment_ages(fishpop& fish) {
  fish.set_ages(fish.get_ages() + 1);
}



