import Image
import numpy as np


def image_as_nparray(image):
    return np.asarray(image)


def nparray_as_image(nparray):
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), 'RGB')
