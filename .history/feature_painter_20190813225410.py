from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import os
import glob

# Neural Network Model #

# Model Methods

OUTPUT_CHANNELS = 3
LAMBDA = 100

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)
    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Input Image Preprocessing

def input_preprocess(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    input_image = tf.cast(image, tf.float32)
    input_image = tf.image.resize(input_image, [256, 256],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / 127.5) - 1
    return input_image

# Program Runner

generator = Generator()
generator.load_weights('machinelearning/gen-weights/ckpt')

def generate(image_file):
    input_image = np.zeros((1, 256, 256, 3))
    input_image[0] = np.array(Image.open(image_file), dtype=np.float32)
    input_image = (input_image / 127.5) - 1
    # generator._make_predict_function()
    prediction = np.array(generator.predict(input_image) * 127.5 + 127.5)
    img = Image.fromarray(prediction[0].astype(np.uint8))
    return img

# Image Methods #

def text(img, text):
    # get an image
    base = img.convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))
    # get a font
    fnt = ImageFont.truetype('arial.ttf', 20)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    # draw text, half opacity
    d.text((32, img.size[1]-32), text, font=fnt, fill=(0,0,0,150))
    out = Image.alpha_composite(base, txt)
    out = out.convert('RGB')
    return out

def paint(filename):
    img = Image.open(latest_file)
    original_size = img.size

    # Start preprocessing image
    img = img.resize((256,256))
    img = img.convert('L')
    img = img.point(lambda x: 255 if x > 128 else 0, '1')
    img = img.convert('RGB')
    img.save('imgs/temp_painter_img.jpg')
    # Paint the image
    img = generate('imgs/temp_painter_img.jpg')
    # Return the processed image to original size
    img = img.resize(original_size)
    img = text(img, "skakmat systems")
    img.save('static/' + latest_file)

# def paint():
#     list_of_files = glob.glob('imgs/*') # * means all if need specific format then *.csv
#     latest_file = max(list_of_files, key=os.path.getctime)
#     img = Image.open(latest_file)
#     original_size = img.size

#     # Start preprocessing image
#     img = img.resize((256,256))
#     img = img.convert('L')
#     img = img.point(lambda x: 255 if x > 128 else 0, '1')
#     img = img.convert('RGB')
#     img.save('imgs/temp_painter_img.jpg')
#     # Paint the image
#     img = generate('imgs/temp_painter_img.jpg')
#     # Return the processed image to original size
#     img = img.resize(original_size)
#     img = text(img, "skakmat systems")
#     img.save('static/' + latest_file)

# paint()