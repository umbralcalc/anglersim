"""
Ecological database class for data obtained from various
open sources. Here are some generally useful links to
datasets and APIs:
https://environment.data.gov.uk/apiportal
https://hub.jncc.gov.uk/assets/52b4e00d-798e-4fbe-a6ca-2c5735ddf049
https://www.cefas.co.uk/data-and-publications/cefas-data-portal-apis/

"""

import os
import json
import requests
import subprocess
import pandas as pd
from io import StringIO


class ecoDB:

    def __init__(self, path : str):
        """
        Class for querying data from various sources.

        Args:
        path
            instantiate the class with the local 
            directory path.
        
        """
        self.path = path
        self._df = None
        self.links = {
            "GB_SAC_shape_file" : "https://data.jncc.gov.uk/"
            + "data/52b4e00d-798e-4fbe-a6ca-2c5735ddf049/"
            + "GB-SAC-OSGB36-20190403.zip",
            # All below from: 
            # https://environment.data.gov.uk/ecology/explorer/downloads/
            # Data documentation (must read!) is here:
            # https://environment.data.gov.uk/ecology/explorer/docs/
            "NFPD_FWfish_counts" : "https://environment.data.gov.uk/"
            + "ecology/explorer/downloads/FW_Fish_Counts.zip",
            "NFPD_FWfish_banded_measurements" : "https://environment"
            + ".data.gov.uk/ecology/explorer/downloads/"
            + "FW_Fish_Banded_Measurements.zip",
            "NFPD_FWfish_bulk_measurements" : "https://environment"
            + ".data.gov.uk/ecology/explorer/downloads/"
            + "FW_Fish_Bulk_Measurements.zip",
            "NFPD_FWfish_data_types" : "https://environment.data.gov.uk/"
            + "ecology/explorer/downloads/FW_Fish_Data_Types.zip",
            "Biosys_FWriver_macroinvertebrates" : "https://environment"
            + ".data.gov.uk/ecology/explorer/downloads/INV_OPEN_DATA.zip",
            "Biosys_FWriver_macrophytes" : "https://environment.data.gov"
            + ".uk/ecology/explorer/downloads/MACP_OPEN_DATA.zip",
            "Biosys_FWriver_diatoms" : "https://environment.data.gov.uk/"
            + "ecology/explorer/downloads/DIAT_OPEN_DATA.zip",
            "Biosys_FWriver_taxon_info" : "https://environment.data.gov"
            + ".uk/ecology/explorer/downloads/OPEN_DATA_TAXON_INFO.zip",
        }
        self.unzipped_files = {
            "GB_SAC_shape_file" : [
                "GB_SAC_OSGB36_20191031.shp",
                "GB_SAC_OSGB36_20191031.shx",
            ],
            "NFPD_FWfish_counts" : ["FW_Fish_Counts.csv"],
            "NFPD_FWfish_banded_measurements" : [
                "FW_Fish_Banded_Measurements.csv",
            ],
            "NFPD_FWfish_bulk_measurements" : [
                "FW_Fish_Bulk_Measurements.csv",
            ],
            "NFPD_FWfish_data_types" : [
                "FW_Fish_Data_Types.csv",
            ],
            "Biosys_FWriver_macroinvertebrates" : [
                "INV_OPEN_DATA_METRICS.csv",
                "INV_OPEN_DATA_SITE.csv",
                "INV_OPEN_DATA_TAXA.csv",
            ],
            "Biosys_FWriver_macrophytes" : [
                "MACP_OPEN_DATA_METRICS.csv",
                "MACP_OPEN_DATA_SITE.csv",
                "MACP_OPEN_DATA_TAXA.csv",
            ],
            "Biosys_FWriver_diatoms" : [
                "DIAT_OPEN_DATA_METRICS.csv",
                "DIAT_OPEN_DATA_SITE.csv",
                "DIAT_OPEN_DATA_TAXA.csv",
            ],
            "Biosys_FWriver_taxon_info" : [
                "OPEN_DATA_TAXON_INFO.csv",
            ],
        }

    def get_unzipped_files(self, name : str):
        """
        Download the zipped folder to the data folder if
        it doesn't already exist and then unzip it into
        its appropriate files.

        Args:
        name
            string name of the data set in self.links.keys().

        """

        if not os.path.exists(
            self.path + "data/" + name + ".zip"
        ):
            bashCommand = "wget " + self.links[name]
            bashCommand += " -O data/" + name + ".zip"
            process = subprocess.Popen(
                bashCommand.split(), 
                cwd=self.path,
            )
            output, error = process.communicate()

        # Unzip the relevant files
        bashCommand = "unzip -o " + name + ".zip"
        for uzf in self.unzipped_files[name]:
            bashCommand += " " + uzf
        process = subprocess.Popen(
            bashCommand.split(), 
            cwd=self.path+'data/',
        )
        output, error = process.communicate()
        