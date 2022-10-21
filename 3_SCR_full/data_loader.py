import os
import math
import random
from PIL import Image
from glob import glob
import tensorflow as tf

def get_loader(config, root, batch_size, data_format, split=None, is_grayscale=False, seed=None, use_neighbor=False, path_block_size = 64):
    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
          w, h = img.size

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=True, seed=seed)
    reader = tf.WholeFileReader()

    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    ##Extract random window from the image
    ################################################################
    BLOCK_SIZE = path_block_size


    image_height, image_width, _ = image.get_shape().as_list()
    number_of_blocks_w = int(math.floor(w / BLOCK_SIZE))
    number_of_blocks_h = int(math.floor(h / BLOCK_SIZE))
    x_idx = random.randint(0, (number_of_blocks_w-1))
    y_idx = random.randint(0, (number_of_blocks_h-1))
    x_offset = x_idx * BLOCK_SIZE
    y_offset = y_idx * BLOCK_SIZE

    image = tf.image.crop_to_bounding_box(image, y_offset, x_offset, BLOCK_SIZE, BLOCK_SIZE)
    ################################################################

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)

    min_after_dequeue = 3000
    capacity = min_after_dequeue + 30 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=8, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    return tf.to_float(queue)
