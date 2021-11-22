"""
Plotting utilities for the NFPD fish dataset.

"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# For nice-looking plots
sns.set()


def plot_fish_counts(
    df : pd.DataFrame,
    mask_dict : dict = None,
    smooth_scale : int = 10,
):
    """
    Plot the count data for the NFPD fish records
    using an arbitrarily specified subset of the 
    input DataFrame.

    Args:
    df
        DataFrame to plot with.

    Keywords:
    mask_dict
        dictionary of column names (as keys)
        and lists (as values) corresponding to
        the desired subset of the DataFrame.
    smooth_scale
        the smoothing scale for the flat window
        moving average to be calculated over.

    """

    # ALL_RUNS is the total of the 6 non-zero (NaN) runs
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'])

    # Create masked DataFrame for plotting based on input
    dfc = df.copy()
    if mask_dict is not None:
        for c, v in mask_dict.items():
            dfc = dfc[dfc[c].isin(v)]

    # Plot raw counts
    dfc.sort_values('EVENT_DATE').set_index('EVENT_DATE')[
        ['RUN1', 'RUN2', 'RUN3', 'RUN4', 'RUN5', 'RUN6']
    ].plot(ax=ax, style='.')

    # Compute the mean estimate and plot the smoothed
    # version using a rolling centered flat window
    dfc['NON_NAN_RUNS'] = (
        dfc['RUN1'].notna()
        + dfc['RUN2'].notna()
        + dfc['RUN3'].notna()
        + dfc['RUN4'].notna()
        + dfc['RUN5'].notna()
        + dfc['RUN6'].notna()
    )
    dfc['MEAN_RUNS'] = dfc['ALL_RUNS'] / dfc['NON_NAN_RUNS']
    (
        dfc
        .sort_values('EVENT_DATE')
    ).set_index('EVENT_DATE')['MEAN_RUNS'].rolling(
        smooth_scale, min_periods=1, center=True
    ).mean().plot(ax=ax, color='k', style='-')
    ax.legend()
    plt.show()