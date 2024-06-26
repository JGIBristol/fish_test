Localisation
====
The first step in the pipeline will be to isolate where the fish's jaw is.

Methods
----
We can do this in several ways - in order of increasing complexity, we can:
 - Just look for a large region of the highest contrast
 - Try to use a template to match where a known bone of interest is (probably the otoliths, as they are very high contrast)
 - Train a neural network to learn where the centre of the jaw is from labelled data

The first method is attempted in `naive.ipynb`; the second in `template_matching.ipynb`; and the third in `neural_net.ipynb` (these may not all exist).

Data
----
Because the `.tiff` images are very large and we don't actually need all that information when we're trialling our methods, I've written a script that
will downsample the images (in 3d) and saves them as numpy arrays.
This can be found in `downsample.py`; it requires the RDSF to be mounted in the directory provided in `userconf.yml`.

Of course the final method will need to be tested on a full-resolution image.
