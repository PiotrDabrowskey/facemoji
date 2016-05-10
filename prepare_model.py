import Image
import glob
import math
import random
import numpy as np
from face_detect import detect_faces

import cv2

fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier


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
        photos = glob.glob('data/sorted_set/%s/*' % emotion)
        filenumber = 1

        for photo in photos:
            image = Image.open(photo)
            frame = cv2.imread(photo)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(frame)

            for (x, y, w, h) in faces:  # get coordinates and size of rectangle containing face
                gray = gray[y:y + h, x:x + w]  # Cut the frame to size

                try:
                    out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                    cv2.imwrite("data/processed/%s/%s.jpg" % (emotion, filenumber), out)  # Write image
                except:
                    print('error in processing %s' %photo)

            filenumber += 1  # Increment image number

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("data/processed/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / (correct + incorrect))

if __name__ == '__main__':
    emotions = ['neutral', 'happy', 'sadness']
    #process_photos(emotions)

    metascore = []
    for i in range(0, 10):
        correct = run_recognizer()
        print "got", correct, "percent correct!"
        metascore.append(correct)

    print "\n\nend score:", np.mean(metascore), "percent correct!"

    fishface.save('models/emotion_detection_model.xml')
