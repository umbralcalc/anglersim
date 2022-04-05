from dataclasses import dataclass
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.typing.simple import *
from src.typing.compound import *


@dataclass
class FWFishCountPlotConfig(PlotConfig):
    def set_user_settings(self):
        species = list(self._df['species.name'].unique())
        species_selection = st.selectbox("Species name", species)
        self._df = self._df[self._df['species.name'].isin([species_selection])]
        self._smooth_scale = st.number_input("Smoothing scale", 0, 100, 10)

    def create_plot(self):
        dat = self._df.sort_values("counts.date").set_index("counts.date")
        dat['counts.total'] = 0.0
        dat['counts.notnans'] = 0.0
        for i in range(1, 7):
            name = "counts." + str(i)
            self.fig.add_trace(
                go.Scatter(
                    x=dat.index,
                    y=dat[name].values,
                    mode="markers",
                    name=name,
                ),
            )
            dat.loc[~(dat[name].isna()), 'counts.total'] += (
                dat.loc[~(dat[name].isna()), name]
            )
            dat['counts.notnans'] += dat[name].notna()
        dat['counts.mean'] = dat['counts.total'] / dat['counts.notnans']
        smoothed_line = (
            dat['counts.mean'].rolling(
                self._smooth_scale, min_periods=1, center=True
            ).mean()
        )
        self.fig.add_trace(
            go.Scatter(
                x=smoothed_line.index,
                y=smoothed_line.values,
                name='counts.smoothed_mean',
                line=dict(color='black'),
            ),
        )

