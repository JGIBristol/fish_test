"""
Misc utilities

"""

import pathlib

import yaml


def config() -> dict:
    """
    Read the config file

    """
    with open("config.yml") as file:
        return yaml.safe_load(file)


def userconf() -> dict:
    """
    Read the user configuration file

    """
    with open(str(pathlib.Path(__file__).parents[1] / "userconf.yml")) as file:
        return yaml.safe_load(file)
