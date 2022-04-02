import datetime as dt
import pandas as pd

from src.data.utils import UNZIPPED_FILES
from src.data.database import get_unzipped_files
from src.typing.simple import *
from src.typing.compound import *


def retrieve_fish_counts() -> FWFishCounts:
    # Retrieve the fish counts data
    get_unzipped_files("NFPD_FWfish_counts")
    df = pd.read_csv(
        "data/" + UNZIPPED_FILES["NFPD_FWfish_counts"][0]
    )
    sites, species, surveys, counts = [], [], [], []
    for _, row in df.iterrows():
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
