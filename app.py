import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response, redirect, url_for
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import random


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")

@app.route("/home")
@nocache
def quiz():
    return render_template('home.html')

@app.route("/about")
@nocache
def about():
    return render_template('about.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
        return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)
    else:
        image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
        return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)
    


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

# start: Filter Spasial

@app.route("/identity", methods=["POST"])
@nocache
def identity():
    image_processing.identity()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/meanBlur2D", methods=["POST"])
@nocache
def meanBlur2D():
    kernel = int(request.form['kernel'])
    image_processing.meanBlur2D(kernel)
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/meanBlurCV", methods=["POST"])
@nocache
def meanBlurCV():
    kernel = int(request.form['kernel'])
    image_processing.meanBlurCV(kernel)
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/gaussian", methods=["POST"])
@nocache
def gaussian():
    kernel = int(request.form['kernel'])
    image_processing.gaussianBlur(kernel)
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/median", methods=["POST"])
@nocache
def median():
    kernel = int(request.form['kernel'])
    image_processing.medianBlur(kernel)
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/sharp", methods=["POST"])
@nocache
def sharp():
    image_processing.sharp()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/bilateral-filter", methods=["POST"])
@nocache
def bilateral():
    image_processing.bilateralFilter()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/zero-padding", methods=["POST"])
@nocache
def zero_padding():
    image_processing.zero_padding()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/lowpass-filter", methods=["POST"])
@nocache
def lowpass():
    kernel = int(request.form['kernel'])
    image_processing.lowFilterPass(kernel)
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/highpass-filter", methods=["POST"])
@nocache
def highpass():
    image_processing.highFilterPass()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/bandpass-filter", methods=["POST"])
@nocache
def bandpass():
    image_processing.bandFilterPass()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

# end: Filter Spasial

@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"], image_dim = image_dimension)
    else:
        image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"], image_dim = image_dimension)


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("uploaded.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route('/create_puzzle', methods=['POST'])
def create_puzzle():
    pieces = int(request.form.get('pieces'))
    image_processing.create_puzzle(pieces)
    return render_template("crop.html", file_path="img/img_now.jpg", pieces=pieces)


@app.route('/random_puzzle', methods=['POST'])
def random_puzzle():
    pieces = int(request.form.get('pieces'))
    image_processing.random_puzzle(pieces)
    return render_template("crop.html", file_path="img/img_now.jpg", pieces=pieces)

@app.route("/get_image_data", methods=["POST"])
@nocache
def get_image_data():
    target = os.path.join(APP_ROOT, "static/img")
    image_dimensions = image_processing.get_image_dimensions("static/img/img_now.jpg")
    rgb_values = image_processing.get_image_rgb("static/img/img_now.jpg")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    return render_template("rgb_values.html", file_path="img/img_now.jpg", image_dim=image_dimensions, img_rgb_val=rgb_values)

# QUIZ COCOKI

@app.route('/play_game', methods=['POST'])
def play_game():
    image_processing.memoryGame()

    photo_folder = 'static/img/memory_game'
    photos = os.listdir(photo_folder)

    random.shuffle(photos)
    photo_data = [{'id': idx, 'filename': photo} for idx, photo in enumerate(photos)]

    random.shuffle(photos)
    photo_data2 = [{'id': idx, 'filename': photo} for idx, photo in enumerate(photos)]

    return render_template('memory_game.html', photos=photo_data, photos2=photo_data2, photo_folder=photo_folder)

# =================================================== CITRA BINER ======================================================
@app.route("/convert2binary", methods=["POST"])
@nocache
def convert2binary():
    image_processing.convertToBinary()
    image_dimension = image_processing.get_image_dimensions('static/img/binary_image.jpg')
    return render_template("binary.html", file_path="img/binary_image.jpg", image_dim = image_dimension)


@app.route("/opening", methods=["POST"])
@nocache
def opening():
    image_processing.opening()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/closing", methods=["POST"])
@nocache
def closing():
    image_processing.closing()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/dilasi", methods=["POST"])
@nocache
def dilasi():
    image_processing.dilasi()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/erosi", methods=["POST"])
@nocache
def erosi():
    image_processing.erosi()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/opening_closing", methods=["POST"])
@nocache
def opening_closing():
    image_processing.opening_closing()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/boundary_extraction", methods=["POST"])
@nocache
def boundary_extraction():
    image_processing.boundary_extraction()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/erosi_slash", methods=["POST"])
@nocache
def erosi_slash():
    image_processing.erosi_slash()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/erosi_backslash", methods=["POST"])
@nocache
def erosi_backslash():
    image_processing.erosi_backslash()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/erosi_dot", methods=["POST"])
@nocache
def erosi_dot():
    image_processing.erosi_dot()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

@app.route("/erosi_stripe", methods=["POST"])
@nocache
def erosi_stripe():
    image_processing.erosi_stripe()
    image_dimension = image_processing.get_image_dimensions('static/img/img_now.jpg')
    return render_template("binary.html", file_path="img/img_now.jpg", image_dim = image_dimension)

# =================================================== KONTUR ======================================================
@app.route("/contour", methods=["POST"])
@nocache
def createNumberImage():
    number = str(request.form.get('number'))
    image_processing.createNumberImage(number)
    file_path = f"img/contour/angka_now.png"
    image_dimension = image_processing.get_image_dimensions("static/" + file_path)
    
    return render_template("contour.html", file_path=file_path, number=number, image_dim = image_dimension)


@app.route("/contour/result", methods=["POST"])
@nocache
def showCheckResult():
    image_dimension = image_processing.get_image_dimensions('static/img/contour/angka_now.png')
    file_path = f"img/contour/angka_now.png"
    result, chain_code = image_processing.showCheckResult()
    return render_template("contour.html", file_path=file_path, image_dim = image_dimension, result=result, chain_code= chain_code)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")



