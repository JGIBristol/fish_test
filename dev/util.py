"""
Misc utilities

"""
import yaml
from functools import cache

@cache
def config() -> dict:
    """
    Read the config file

    """
    with open("config.yml") as file:
        return yaml.safe_load(file)