"""
Crop out the fish head by looking at the greyscale intensity of the image and making some assumptions about the fish's orientation

"""


def main():
    """
    Read in the image, count how many white pixels there are for various thresholds and take the average

    Then find peaks in the allowed region (not near the edges, profile decreasing, not too small) and
    choose one of these peaks

    Then from the corresponding image slice, find the x/y window that contains the head and crop it out

    """
    # Read in the image
    # For various thresholds, count the number of white pixels
    # Find peaks
    # Define our allowed regions and find where peaks are allowed
    # Choose a peak and find the corresponding slice
    # Choose the x/y window

if __name__ == "__main__":
    main()
