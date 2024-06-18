"""
Crop arrays according to the centres in the mastersheet

"""

from dev import metadata

import pydicom


def main():
    """
    For each stack of tifs, read the mastersheet to find the jaw centre, read the right images and crop them
    Then save them as a DICOM

    """
    # Read the relevant columns in the mastersheet
    mastersheet = metadata.mastersheet()[["old_n", "jaw_center"]].set_index("old_n")

    # Choose a directory for images to be saved

    # For each image, read and crop the central region, then save to DICOM


if __name__ == "__main__":
    main()
