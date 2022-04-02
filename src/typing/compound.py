from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import datetime as dt

from src.typing.simple import *



@dataclass
class Species:
    id: str
    name: str


@dataclass
class Survey:
    id: str
    method: str
    strategy: str
    length: float
    width: float
    area: float
    species_selective: bool # important one here!
    third_party: bool


@dataclass
class Site:
    id: str
    name: str
    location_name: str
    region: str
    country: str
    geo_waterbody: str


@dataclass
class Counts:
    by_run: Dict[Run, RunCount]
    date: dt.datetime
    


@dataclass
class PlottableData:
    """base class to retrieve plotting data"""

    def to_df(self) -> pd.DataFrame: 
        return pd.DataFrame.from_dict(asdict(self))


@dataclass
class PlotConfig:
    """base class for plot configs"""
    data: PlottableData
    fig: Optional[go.Figure]

    def __post_init__(self):
        self.fig = go.Figure()

    def set_user_settings(self):
        """call streamlit settings routine
        and set config attributes"""
        pass

    def create_plot(self):
        """create the actual plot"""


@dataclass
class FWFishCounts(PlottableData):
    sites: List[Site]
    species: List[Species]
    surveys: List[Survey]
    counts: List[Counts]



@dataclass
class AppConfig:
    mode: AppModes
    fish_count_data: FWFishCounts