import Image
import glob
import math
import numpy as np
from sklearn.metrics import accuracy_score

import cv2


def distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def scale_rotate_translate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def crop_face(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])  # calculate offsets in original image
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])  # get the direction
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))  # calc rotation angle in radians
    dist = distance(eye_left, eye_right)  # distance between them
    reference = dest_sz[0] - 2.0 * offset_h  # calculate the reference eye-width
    scale = float(dist) / float(reference)  # scale factor
    image = scale_rotate_translate(image, center=eye_left, angle=rotation)  # rotate original around the left eye
    xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)  # crop the rotated image
    size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop((int(xy[0]), int(xy[1]), int(xy[0] + size[0]), int(xy[1] + size[1])))
    image = image.resize(dest_sz, Image.ANTIALIAS)  # resize it
    return image


def process_photos(emotions):
    for emotion in emotions:
        photos = glob.glob('data/%s/*' % emotion)
        index = 1
        for photo in photos:
            image = Image.open(photo)
            cords = map(int, photo.split('/')[2].split('.')[0].split(','))
            filename = 'processed/' + emotion + '/' + str(index) + '.' + photo.split('.')[1]
            crop_face(image,
                      eye_left=(int(cords[0]), cords[1]),
                      eye_right=(cords[2], cords[3]),
                      offset_pct=(0.3, 0.3),
                      dest_sz=(200, 200)).save(filename)
            index += 1


def get_dataset(emotions):
    data = []
    labels = []
    for emotion in emotions:
        photos = glob.glob('processed/%s/*' % emotion)
        for item in photos:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            data.append(gray)  # append image array to training data list
            labels.append(emotions.index(emotion))  # add numeric label
    return data, labels


def train_model(data, labels):
    fisher_face = cv2.createFisherFaceRecognizer()  # initialize fisher face classifier
    fisher_face.train(data, np.asarray(labels))  # train model
    return fisher_face


def evaluate_model(data, labels, model):
    predicted = [model.predict(x)[0] for x in data]
    return accuracy_score(labels, predicted)


if __name__ == '__main__':
    emotions = ['happy', 'sad']
    process_photos(emotions)

    data, labels = get_dataset(emotions)
    model = train_model(data, labels)

    accuracy = evaluate_model(data, labels, model)
    print('trained model accuracy (on train data): {}'.format(accuracy))

    model.save('emotion_detection_model.xml')
