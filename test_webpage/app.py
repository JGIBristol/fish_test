import os
from tempfile import SpooledTemporaryFile

import numpy as np
from scipy import ndimage
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session


app = Flask(__name__)

# Ensure there's a folder to save the uploaded files
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def process(img_file: SpooledTemporaryFile, rotation: int = 0) -> str:
    """
    Do some stuff with the file, return path to it

    """
    img_file = Image.open(img_file.stream)
    array = np.array(img_file)

    rotated = Image.fromarray(ndimage.rotate(array, angle=rotation))

    # Save the rotated image
    filename = f"rotated_{rotation}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    rotated.save(filepath)

    return filename


def process_multiple(img_file: SpooledTemporaryFile, rotations: list[int]) -> list[str]:
    return [process(img_file, rotation) for rotation in rotations]


@app.route("/")
def index():
    return render_template("upload_form.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return redirect(request.url)
    file = request.files["image"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        filenames = process_multiple(file, [-60, -30, 30, 60])

        return render_template("display_image.html", filenames=filenames)


@app.route("/display")
def display_image():
    filenames = session.get("filenames", [])
    return render_template("display_image.html", filenames=filenames)


if __name__ == "__main__":
    app.run(debug=True)
