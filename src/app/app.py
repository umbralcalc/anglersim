from dataclasses import dataclass
import streamlit as st

from src.app.plotters import FWFishCountPlotConfig, PlotConfig

from src.typing.simple import *
from src.typing.compound import *


@dataclass
class App:
    config: AppConfig

    def _run(self, plot_config: PlotConfig):
        plot_config.set_user_settings()
        plot_config.create_plot()

    def run(self):
        if self.config.mode == AppModes.data_plotter:
            self._run(
                FWFishCountPlotConfig(
                    data=self.config.fish_count_data,
                )
            )
