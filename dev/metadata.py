"""
Fish stuff metadata

"""

import os
import requests
from functools import cache

import pandas as pd


def download_mastersheet() -> None:
    """
    Download the mastersheet from GitHub

    """
    url = (
        "https://raw.githubusercontent.com/wahabk/ctfishpy/master/ctfishpy/Metadata/uCT_mastersheet.csv",
    )
    response = requests.get(url)
    response.raise_for_status()

    with open("uCT_mastersheet.csv", "wb") as file:
        file.write(response.content)


@cache
def mastersheet() -> pd.DataFrame:
    """
    Download the mastersheet from GitHub if necessary and return it as a DataFrame

    """
    if not os.path.exists("uCT_mastersheet.csv"):
        download_mastersheet()
    return pd.read_csv("uCT_mastersheet.csv")
