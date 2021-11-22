"""
Database query utils for data obtained from the
data.gov API.

"""

import json
import requests
import pandas as pd
from io import StringIO


def get_all_govuk_dataset_names() -> list:
    """Get the available data.gov dataset names as a list."""
     
    x = requests.get(
        'https://data.gov.uk/api/action/package_list'
    )
    return x.json()['result']


def search_govuk_dataset_refs(
    search_string : str,
    print_to_screen=True
) -> dict:
    """
    Search the available data.gov datasets and get
    link references as a dictionary.
    
    Args:
    search_string
        String to search the datasets for.

    Keywords:
    print_to_screen
        Boolean to print results or not.

    """
     
    x = requests.get(
        'https://data.gov.uk/api/action/package_search?q='
        + search_string
    )
    y = x.json()['result']['results']
    dict_of_links = {}
    for v in y:
        for vv in v['resources']:
            dict_of_links[vv['name']] = vv['url']
            if print_to_screen:
                print(vv['name'])
                print(vv['url'])

    return dict_of_links
