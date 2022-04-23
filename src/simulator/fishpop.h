
// Fish population representation in the simulation.
/// @file    fishpop.h

#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

/// @class fishpop
/// @brief Representation of a whole fish population.
class fishpop
{
  xt::xarray<int> ages;
  xt::xarray<int> counts;
  xt::xarray<double> weights;

  public:
    void create(
      xt::xarray<int> init_ages,
      xt::xarray<int> init_counts,
      xt::xarray<double> init_weights
    );
    void set_ages(xt::xarray<int> new_ages);
    void set_counts(xt::xarray<int> new_counts);
    void set_weights(xt::xarray<double> new_weights);
    xt::xarray<int> get_ages();
    xt::xarray<int> get_counts();
    xt::xarray<double> get_weights();
};
