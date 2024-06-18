"""
Read, crop, save, etc. images

"""

import yaml
import pathlib
from functools import cache


@cache
def _user_config() -> dict:
    """
    Read the user configuration file

    """
    with open("userconf.yml") as file:
        return yaml.safe_load(file)


@cache
def ct_scan_dir() -> pathlib.Path:
    """
    Path to the directory

    """
    return (
        pathlib.Path(_user_config()["rdsf_dir"])
        / "DATABASE"
        / "uCT"
        / "Wahab_clean_dataset"
        / "low_res_clean_v3"
    )


def img_dir(img_n: int) -> pathlib.Path:
    """
    Path to the directory of the image

    :param img_n: "old_n" in the mastersheet

    """
    return ct_scan_dir() / str(img_n) / "reconstructed_tifs"
