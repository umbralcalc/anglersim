"""
Plotting utilities for the NFPD fish dataset.

"""

import datetime as dt
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt

from src.data.utils import UNZIPPED_FILES
from src.data.database import get_unzipped_files


class Run(str):
    def __new__(cls, *args, **kwargs):
        return super(Run, str).__new__(cls, *args, **kwargs)


class RunCount(int):
    def __new__(cls, *args, **kwargs):
        return super(RunCount, int).__new__(cls, *args, **kwargs)


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
class FWFishCounts:
    sites: List[Site]
    species: List[Species]
    surveys: List[Survey]
    counts: List[Counts]



def retrieve_fish_counts():
    # Retrieve the fish counts data
    get_unzipped_files("NFPD_FWfish_counts")
    df = pd.read_csv(
        "data/" + UNZIPPED_FILES["NFPD_FWfish_counts"][0]
    )
    sites, species, surveys, counts = [], [], [], []
    for row in df.iterrows():
        sites.append(
            Site(
                id=row.SITE_ID,
                name=row.SITE_NAME,
                location_name=row.LOCATION_NAME,
                region=row.REGION,
                country=row.COUNTRY,
                geo_waterbody=row.GEO_WATERBODY,
            )
        )
        surveys.append(
            Survey(
                id=row.SURVEY_ID,
                method=row.SURVEY_METHOD,
                strategy=row.SURVEY_STRATEGY,
                length=row.SURVEY_LENGTH,
                width=row.SURVEY_WIDTH,
                area=row.SURVEY_AREA,
                species_selective=row.IS_SPECIES_SELECTIVE,
                third_party=row.IS_THIRD_PARTY,
            )
        )
        species.append(Species(id=row.SPECIES_ID, name=row.SPECIES_NAME))
        by_run = {
            Run(str(i)) : RunCount(row["RUN" + str(i)])
            for i in range(1, 7)
        }
        counts.append(Counts(by_run=by_run, date=dt.datetime(row.EVENT_DATE)))
    return FWFishCounts(
        sites=sites,
        species=species,
        surveys=surveys,
        counts=counts,
    )


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