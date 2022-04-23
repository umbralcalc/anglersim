
// The dynamical population model driving the simulation.
/// @file    dynamicalpop.h

#pragma once
#include "fishpop.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>



/// @class dynamicalpop
/// @brief An interface for the fish populations to evolve in the simulation.
class dynamicalpop
{
  xt::xarray<int> time;

  public:
    void increment_ages(fishpop& fish);
};
