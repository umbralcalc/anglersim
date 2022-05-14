from dataclasses import dataclass
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
        ax = self.fig.add_subplot()
        for i in range(1, 7):
            name = "counts." + str(i)
            ax.scatter(
                x=dat.index,
                y=dat[name].values,
                label=name,
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
        ax.plot(
            smoothed_line.index,
            smoothed_line.values,
            label='counts.smoothed_mean',
            color='k',
        )

