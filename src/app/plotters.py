from dataclasses import dataclass
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging

from src.typing.simple import *
from src.typing.compound import *


@dataclass
class FWFishCountPlotConfig(PlotConfig):
    def set_user_settings(self):
        """implement this"""

    def create_plot(self):
        dfs = self.data.to_dfs()
        logging.info("retrieved the fish counts plot data")

        """

        # Create masked DataFrame for plotting based on input
        dfc = df.copy()
        if mask_dict is not None:
            for c, v in mask_dict.items():
                dfc = dfc[dfc[c].isin(v)]

        """

        # Plot raw counts
        dat = dfs["counts"].sort_values("date").set_index("date")
        for i in range(1, 7):
            name = str(i)
            self.fig.add_trace(
                go.Scatter(
                    x=dat.index,
                    y=dat[name].values,
                    mode="markers",
                    name=name,
                ),
            )
        st.plotly_chart(self.fig)

        """

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

        """
