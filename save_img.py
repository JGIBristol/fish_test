"""
Cache the images to disk as numpy arrays

"""

import argparse

from dev import image_io


def main(img_n: int):
    """Read the selected image"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("img_n", type=int, help="The image number")
    main(*parser.parse_args())
