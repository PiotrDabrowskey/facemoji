import Image
import glob
import numpy as np
import random

import cv2

from align_photos import crop_face
from face_detect import detect_face_and_crop
from image_converters import nparray_to_image, image_to_nparray

emotions = ["happy", "sad"]  # Emotion list
fishface = cv2.createFisherFaceRecognizer()  # Initialize fisher face classifier

data = {}


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("processed/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # repeat above process for prediction set
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


def transform_webcam():
    # initialize the camera
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        face_from_frame = detect_face_and_crop(frame)
        # face_from_frame = crop_face(nparray_to_image(frame),
        #                             eye_left=(int(coordinates[0]), coordinates[1]),
        #                             eye_right=(coordinates[2], coordinates[3]),
        #                             offset_pct=(0.3, 0.3), dest_sz=(200, 200)
        #                             )
        if face_from_frame is not None:
            print 'detected'
            cv2.imshow('face', face_from_frame)
        else:
            print 'not detected'

        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")


# Now run it
# metascore = []
# for i in range(0, 10):
#     correct = run_recognizer()
#     print "got", correct, "percent correct!"
#     metascore.append(correct)
#
# print "\n\nend score:", np.mean(metascore), "percent correct!"

transform_webcam()
