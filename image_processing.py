import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2
import random
import os
import pandas as pdpip
from skimage import io
matplotlib.use("Agg")
import scipy.ndimage.morphology as m
# memasukkan hasil chain code ke dalam .env
from dotenv import dotenv_values, set_key

def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)

    if len(img_arr.shape) == 2:
        new_img = Image.fromarray(img_arr)
    else:
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        new_arr = r.astype(int) + g.astype(int) + b.astype(int)
        new_arr = (new_arr/3).astype('uint8')
        new_img = Image.fromarray(new_arr)

    new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True


def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")
    


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def convolution(img, kernel):
    h_img, w_img = img.shape[:2]  
    out = np.zeros((h_img - 2, w_img - 2), dtype=np.float32)
    
    if len(img.shape) == 3:  
        new_img = np.zeros((h_img - 2, w_img - 2, 3))
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img - 2):
                for w in range(w_img - 2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
        new_img = np.uint8(new_img)
    else: 
        array = img
        for h in range(h_img - 2):
            for w in range(w_img - 2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        new_img = np.uint8(out_)
    return new_img

def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype= np.int_)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int_)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int_)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def save_histogram(data, color):
    fig, ax = plt.subplots()
    ax.bar(list(data.keys()), data.values(), color=color)
    plt.savefig(f'static/img/{color}_histogram.jpg', dpi=300)
    plt.clf()
    plt.close()

def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)

    if len(img_arr.shape) == 2:
        # Grayscale image
        data_g = Counter(img_arr.flatten())
        matplotlib.use('Agg')
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    elif len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        # Color image (assuming it's RGB)
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()

        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)

        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        matplotlib.use('Agg')
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()
    else:
#         Handle other cases or raise an error if the image format is not supported
        raise ValueError("Unsupported image format")
    


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)

def threshold(lower_thres, upper_thres):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))
    print(lower_thres, upper_thres)
    
    # Membuat salinan array yang dapat diubah
    img_arr_copy = img_arr.copy()
    img_arr_copy[condition] = 255
    
    new_img = Image.fromarray(img_arr_copy)
    new_img.save("static/img/img_now.jpg")


def create_puzzle(pieces):
    # Pastikan pieces adalah angka
    if not isinstance(pieces, int) or pieces <= 0:
        raise ValueError("Parameter 'pieces' harus menjadi bilangan bulat positif")

    # Baca gambar
    image = Image.open("static/img/img_now.jpg")
    width, height = image.size

    cols = rows = pieces  # Gunakan nilai yang sama untuk cols dan rows

    # Hitung ukuran potongan
    piece_width = width // cols
    piece_height = height // rows

    for row in range(rows):
        for col in range(cols):
            # Hitung koordinat potongan
            left = col * piece_width
            upper = row * piece_height
            right = left + piece_width
            lower = upper + piece_height

            # Potong gambar
            piece = image.crop((left, upper, right, lower))

            # Simpan potongan ke file
            piece.save(f"static/img/puzzle_pieces/piece_{row}_{col}.png")


def random_puzzle(pieces):
    # Pastikan pieces adalah angka
    if not isinstance(pieces, int) or pieces <= 0:
        raise ValueError("Parameter 'pieces' harus menjadi bilangan bulat positif")

     # Baca gambar
    image = Image.open("static/img/img_now.jpg")
    width, height = image.size

    cols = rows = pieces  # Gunakan nilai yang sama untuk cols dan rows

    # Hitung ukuran potongan
    piece_width = width // cols
    piece_height = height // rows

    # Buat list untuk menyimpan potongan gambar
    pieces = []

    for row in range(rows):
        for col in range(cols):
            # Hitung koordinat potongan
            left = col * piece_width
            upper = row * piece_height
            right = left + piece_width
            lower = upper + piece_height

            # Potong gambar
            piece = image.crop((left, upper, right, lower))

            # Simpan potongan dalam list
            pieces.append(piece)

    # Acak urutan potongan gambar
    random.shuffle(pieces)

    # Simpan potongan ke file dengan nama yang berurutan
    for i, piece in enumerate(pieces):
        row = i // cols
        col = i % cols
        piece.save(f"{'static/img/puzzle_pieces'}/piece_{row}_{col}.png")

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        return None

def get_image_rgb(image_path):
    try:
        with Image.open(image_path) as img:
            rgb_values = list(img.getdata())
            return rgb_values
    except Exception as e:
        return None

def convolution(img, kernel):
    h_img, w_img = img.shape[:2]  
    out = np.zeros((h_img - 2, w_img - 2), dtype=np.float32)
    
    if len(img.shape) == 3:  
        new_img = np.zeros((h_img - 2, w_img - 2, 3))
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img - 2):
                for w in range(w_img - 2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
        new_img = np.uint8(new_img)
    else: 
        array = img
        for h in range(h_img - 2):
            for w in range(w_img - 2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        new_img = np.uint8(out_)
    return new_img

# filter spasial

def identity():
    img = cv2.imread('static\img\img_now.jpg')
    kernel = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])
    
    identity = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imwrite('static/img/img_now.jpg', identity)

def meanBlur2D(kernel):
    img = cv2.imread('static\img\img_now.jpg')
    kernel = np.ones((kernel, kernel), np.float32) / (kernel*kernel)
    blur = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imwrite('static/img/img_now.jpg', blur)

def meanBlurCV(kernel):
    img = cv2.imread("static/img/img_now.jpg")
    cv_blur = cv2.blur(src=img, ksize=(kernel, kernel))
    cv2.imwrite("static/img/img_now.jpg", cv_blur)

def gaussianBlur(kernel):
    img = cv2.imread('static\img\img_now.jpg')
    cv_gaussianblur = cv2.GaussianBlur(src=img,ksize=(kernel,kernel),sigmaX=0)
    cv2.imwrite('static/img/img_now.jpg', cv_gaussianblur)
    
def medianBlur(kernel):
    img = cv2.imread('static\img\img_now.jpg')
    cv_median = cv2.medianBlur(src=img, ksize=kernel)
    cv2.imwrite('static/img/img_now.jpg', cv_median)
    
def sharp():
    img = cv2.imread('static\img\img_now.jpg')
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

    sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imwrite('static/img/img_now.jpg', sharp)
    
def bilateralFilter():
    img = cv2.imread('static\img\img_now.jpg')
    bf = cv2.bilateralFilter(src=img,d=9,sigmaColor=75,sigmaSpace=75)
    cv2.imwrite('static/img/img_now.jpg', bf)
    
def zero_padding():
    img = cv2.imread('static\img\img_now.jpg')
    zeroPadding = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite('static/img/img_now.jpg', zeroPadding)
    
def lowFilterPass(kernel):
    img = cv2.imread('static\img\img_now.jpg')
    # create the low pass filter
    lowFilter = np.ones((kernel,kernel),np.float32)/(kernel*kernel)
    # apply the low pass filter to the image
    lowFilterImage = cv2.filter2D(img,-1,lowFilter)
    cv2.imwrite('static/img/img_now.jpg', lowFilterImage)
    
def highFilterPass():
    img = cv2.imread('static\img\img_now.jpg')
    # create the high pass filter
    highFilter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # apply the high pass filter to the image
    highFilterImage = cv2.filter2D(img,-1,highFilter)
    cv2.imwrite('static/img/img_now.jpg', highFilterImage)
    
def bandFilterPass():
    img = cv2.imread('static\img\img_now.jpg')
    # create the band pass filter
    bandFilter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    # apply the band pass filter to the image
    bandFilterImage = cv2.filter2D(img,-1,bandFilter)
    cv2.imwrite('static/img/img_now.jpg', bandFilterImage)

# QUIZ
def memoryGame():
    image = cv2.imread("static/img/img_normal.jpg")

    # ================== Greyscale ==================
    if is_grey_scale("static/img/img_normal.jpg"):
        return
    else:
        img_arr = np.asarray(image)
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        new_arr = r.astype(int) + g.astype(int) + b.astype(int)
        new_arr = (new_arr/3).astype('uint8')
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/memory_game/img_grayscale.jpg")

    # ================== Move Left ==================
    img_arr = np.asarray(image)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_left.jpg")

    # ================== Move Right ==================
    img_arr = np.asarray(image)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_right.jpg")

    # ================== Move Up ==================
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(image)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_up.jpg")


    # ================== Move Down ==================
    img_arr = np.asarray(image)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_down.jpg")


    # ================== Median Filter ==================
    cv_median = cv2.medianBlur(image, 9)
    cv2.imwrite("static/img/memory_game/img_median.jpg", cv_median)

    # ================== Edge Detection / High Pass Filter ==================
    img = cv2.imread('static\img\img_now.jpg')
    # create the high pass filter
    highFilter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # apply the high pass filter to the image
    highFilterImage = cv2.filter2D(img,-1,highFilter)
    cv2.imwrite('static/img/img_edge_detection.jpg', highFilterImage)

    # ================== Sharpening ==================
    image_0 = io.imread("static/img/img_now.jpg")
    image = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imwrite("static/img/memory_game/img_sharpen.jpg", sharp)

    # ================== Low Pass Filter ==================
    image = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
    lowFilter = np.ones((5,5),np.float32)/9
    lowFilterImage = cv2.filter2D(image,-1,lowFilter)
    cv2.imwrite("static/img/memory_game/img_lowpass.jpg", lowFilterImage)

    # ================== Band Pass Filter ==================
    image = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
    
    lowFilter = np.ones((3, 3), np.float32) / (3 * 3)
    lowFilterImage = cv2.filter2D(image, -1, lowFilter)
    highFilter = np.ones((5, 5), np.float32) / (5 * 5)
    highPassImage = image - cv2.filter2D(lowFilterImage, -1, highFilter)
    cv2.imwrite("static/img/memory_game/img_bandpass.jpg", highPassImage)


   # ================== Brightness Addition ==================
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_brightness_add.jpg")


    # ================== Brightness Substraction ==================
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_brightness_substract.jpg")


    # ================== Brightness Multiplication ==================
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_brightness_multiply.jpg")


   # ================== Brightness Division ==================
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/memory_game/img_brightness_div.jpg")

# =================================================== CITRA BINER ======================================================
# Convert to Binary
def convertToBinary():
    # Baca citra
    image_path = "static/img/img_now.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Metode Otsu untuk menghitung threshold otomatis
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Simpan citra-citra hasil
    cv2.imwrite("static/img/binary_image.jpg", binary_image)

def opening():
    image = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    
    # Dilasi
    dilasi = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Opening (Erosi diikuti oleh dilasi)
    opening = cv2.morphologyEx(dilasi, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite('static/img/img_now.jpg', opening)

def closing():
    image = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    
    # Dilasi
    dilasi = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Closing (Dilasi diikuti oleh erosi)
    closing = cv2.morphologyEx(dilasi, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite('static/img/img_now.jpg', closing)

def erosi():
    image = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform erosion to remove small white spots outside the object
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    
    cv2.imwrite('static/img/img_now.jpg', eroded_image)
    

def dilasi():
    image = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform erosion to remove small white spots outside the object
    dilasi = cv2.dilate(binary_image, kernel, iterations=1)
    
    cv2.imwrite('static/img/img_now.jpg', dilasi)
    
def opening_closing():
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)

    # Dilasi
    dilasi = cv2.dilate(binary_image, kernel, iterations=1)

    # Closing (Dilasi diikuti oleh erosi)
    closing = cv2.morphologyEx(dilasi, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("static/img/img_now.jpg", closing)
    
def boundary_extraction():
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)

    # Perform erosion to remove small white spots outside the object
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)   

    

    cv2.imwrite("static/img/img_now.jpg", eroded_image)

    # Ekstraksi Batas
    boundary = binary_image - eroded_image

    cv2.imwrite("static/img/img_now.jpg", boundary)
    
def erosi_slash():
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create custom kernels to remove slash and backslash erosions
    custom_kernel_slash = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=np.uint8)

    # Apply morphological operations using different structural elements
    slash = cv2.erode(binary_image, custom_kernel_slash, iterations=1) 

    cv2.imwrite("static/img/img_now.jpg", slash)
    
def erosi_backslash():
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create custom kernels to remove slash and backslash erosions
    custom_kernel_backslash = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
    ], dtype=np.uint8)

    # Apply morphological operations using different structural elements
    backslash = cv2.erode(binary_image, custom_kernel_backslash, iterations=1) 

    cv2.imwrite("static/img/img_now.jpg", backslash)
    
def erosi_dot():
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create custom kernels to remove slash and backslash erosions
    custom_kernel_dot = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0]
    ], dtype=np.uint8)

    # Apply morphological operations using different structural elements
    dot1 = cv2.erode(binary_image, custom_kernel_dot, iterations=1)

    cv2.imwrite("static/img/img_now.jpg", dot1)
    
def erosi_stripe():
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    # Ambil ambang biner, misalnya dengan thresholding Otsu
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create custom kernels to remove slash and backslash erosions
    custom_kernel_stripe = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.uint8)

    # Apply morphological operations using different structural elements
    stripe = cv2.erode(binary_image, custom_kernel_stripe, iterations=1)

    cv2.imwrite("static/img/img_now.jpg", stripe)

# =================================================== KONTUR ======================================================

def createNumberImage(number):
    number = str(number)
    # Ukuran font yang diinginkan
    font_scale_desired = 1

    # Membuat citra untuk angka 

    # Membuat citra kosong berwarna putih dengan lebar 100 piksel dan tinggi 100 piksel
    width = 100
    height = 100
    color = (255, 255, 255)  # Putih dalam format BGR
    image = 255 * np.ones((height, width, 3), dtype=np.uint8)

    # Menambahkan angka ke citra
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(number)
    org = (25, 60)  # Posisi teks pada citra
    font_color = (0, 0, 0)  # Hitam dalam format BGR
    font_thickness = 3

    # Perhitungan posisi teks agar berada di tengah
    text_size = cv2.getTextSize(text, font, font_scale_desired, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    org = (text_x, text_y)  # Posisi teks pada citra

    cv2.putText(image, text, org, font, font_scale_desired, font_color, font_thickness)

    # Menyimpan citra ke dalam file
    save_directory = "static/img/contour/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    file_path = os.path.join(save_directory, f"angka_now.png")
    cv2.imwrite(file_path, image)

def showCheckResult():
    # Load Freeman Chain Codes from .env file
    env_file = ".env"
    env_data = dotenv_values(env_file)
    angka_chain_codes = {}
    for key, value in env_data.items():
        if key.startswith("angka_") and key.endswith("_chain_code"):
            angka = key.replace("_chain_code", "")
            angka_chain_codes[angka] = [int(x) for x in value.split(',')]

    # Function to compare chain codes
    def compare_chain_codes(chain_code1, chain_code2):
        if len(chain_code1) != len(chain_code2):
            return float('inf')  # Chain code not matching

        distance = sum((a - b) ** 2 for a, b in zip(chain_code1, chain_code2)) ** 0.5
        return distance

    # Function to detect digits in an image and collect their chain codes
    def detect_digits_in_image(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Sort contours from left to right based on x-coordinate
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        recognized_digits = []
        chain_codes = []

        for contour in contours:
            # Calculate Freeman Chain Code
            chain_code = freeman_chain_code(contour)

            for angka, chain_code_angka in angka_chain_codes.items():
                if ''.join(map(str, chain_code_angka)) in ''.join(map(str, chain_code)):
                    recognized_digits.append(int(angka.split('_')[1]))  # Convert angka string to integer
                    chain_codes.append(chain_code)
            
                    img_with_contours = cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
                    img_rgb = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)

        cv2.imwrite("static/img/contour/angka_kontur.png", img_rgb)

        return recognized_digits, chain_codes

    # Function to compute Freeman Chain Code
    def freeman_chain_code(contour):
        chain_code = []

        # List of Freeman Chain Code directions
        directions = [0, 1, 2, 3, 4, 5, 6, 7]

        # Initial point
        start_point = contour[0][0]
        current_point = start_point

        # Loop to follow the contour line
        for point in contour[1:]:
            x, y = point[0]
            dx = x - current_point[0]
            dy = y - current_point[1]
            direction = None

            # Determine the direction based on coordinate changes
            if dx == 1 and dy == 0:
                direction = 0
            elif dx == 1 and dy == -1:
                direction = 1
            elif dx == 0 and dy == -1:
                direction = 2
            elif dx == -1 and dy == -1:
                direction = 3
            elif dx == -1 and dy == 0:
                direction = 4
            elif dx == -1 and dy == 1:
                direction = 5
            elif dx == 0 and dy == 1:
                direction = 6
            elif dx == 1 and dy == 1:
                direction = 7

            if direction is not None:
                chain_code.append(direction)
                current_point = (x, y)

        return chain_code

    # Run the detection on a sample image
    image_path = 'static/img/contour/angka_now.png'

    img = cv2.imread('static/img/contour/angka_now.png', 0)

    thin_img = skeletonize(img)

    cv2.imwrite('static/img/contour/angka_thin.png', thin_img)

    recognized_digits, chain_codes = detect_digits_in_image(image_path)

    # Mengidentifikasi kontur pada citra thinning
    contours, _ = cv2.findContours(thin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    chain_codes2 = []
    for contour in contours:
        chain_code2 = freeman_chain_code(contour)
        for angka, chain_code_angka in angka_chain_codes.items():
            chain_codes2.append(chain_code2)


    return recognized_digits, chain_codes


def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = 255 - img

    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel
