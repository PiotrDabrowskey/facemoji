import Image, cv2
import numpy as np


def image_as_nparray(image):
    return np.asarray(image)


def nparray_as_image(nparray, mode='RGB'):
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)

def load_gray_image(source):
    image = cv2.imread(source)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray