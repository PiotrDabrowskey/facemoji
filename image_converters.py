import cv2
if cv2.__version__ == '3.1.0':
    from PIL import Image
else:
    import Image
import numpy as np

def image_as_nparray(image):
    return np.asarray(image)

def nparray_as_image(nparray, mode='RGB'):
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)

def load_image(source):
    image = cv2.imread(source)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)