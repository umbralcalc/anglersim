from dataclasses import Field, dataclass, asdict, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import plotly.graph_objects as go
import datetime as dt

from src.typing.simple import *


class DataframeableData:
    """base class for data which can turn into dataframes"""

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(asdict(self))


@dataclass
class Species(DataframeableData):
    id: List[str]
    name: List[str]


@dataclass
class Surveys(DataframeableData):
    id: List[str]
    method: List[str]
    strategy: List[str]
    length: List[float]
    width: List[float]
    area: List[float]
    species_selective: List[bool]  # important one here!
    third_party: List[bool]


@dataclass
class Sites(DataframeableData):
    id: List[str]
    name: List[str]
    location_name: List[str]
    region: List[str]
    country: List[str]
    geo_waterbody: List[str]


@dataclass
class Counts:
    by_run: Dict[Run, List[Union[int, Any]]]
    date: List[dt.datetime]

    def to_df(self) -> pd.DataFrame:
        dat = {str(k): v for k, v in self.by_run.items()}
        dat["date"] = self.date
        return pd.DataFrame.from_dict(dat)


@dataclass
class PlottableData:
    """base class to retrieve plotting data"""

    def to_dfs(self) -> Dict[str, pd.DataFrame]:
        cls_fields: Tuple[Field, ...] = fields(self.__class__)
        return {
            field.name : getattr(getattr(self, field.name), "to_df")()
            for field in cls_fields
        }


@dataclass
class PlotConfig:
    """base class for plot configs"""

    data: PlottableData
    fig: Optional[go.Figure] = None

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
    sites: Sites
    species: Species
    surveys: Surveys
    counts: Counts


@dataclass
class AppConfig:
    mode: AppModes
    fish_count_data: FWFishCounts
