"""
Misc utilities

"""
import pathlib

import yaml
from functools import cache


@cache
def config() -> dict:
    """
    Read the config file

    """
    with open("config.yml") as file:
        return yaml.safe_load(file)


@cache
def userconf() -> dict:
    """
    Read the user configuration file

    """
    with open(str(pathlib.Path(__file__).parents[1] / "userconf.yml")) as file:
        return yaml.safe_load(file)
