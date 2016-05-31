"""
This file contains logic responsible for teaching model on a processed dataset. It prints it's accuracy.
Based on Paul van Gent's code from blog post: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
"""
import glob
import random
import numpy as np
import cv2

from image_commons import load_image

if cv2.__version__ != '3.1.0':
    fishface = cv2.createFisherFaceRecognizer()
else:
    fishface = cv2.face.createFisherFaceRecognizer()
training_set_size = 0.95


def get_files(emotion):
    """
    gets paths to all images of given emotion and splits them into two sets: trainging and test
    :param emotion: name of emotion to find images for
    """
    files = glob.glob("data/sorted_set/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]
    return training, prediction


def make_sets():
    """
    method used to create datasets for all emotions. It loads both images and its labels to memory into training and test labels
    """
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)

        for item in training:
            training_data.append(load_image(item))
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            prediction_data.append(load_image(item))
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    """
    method is creating datasets using make_sets method, then it trains a model and tet with a test set. It returns correct guesses to test data count ratio
    """
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))

    print("predicting classification set")
    correct = sum(1 for id, image in enumerate(prediction_data) if fishface.predict(image)[0] == prediction_labels[id])

    return ((100 * correct) / len(prediction_data))


if __name__ == '__main__':
    emotions = ["neutral", "anger", "disgust", "happy", "sadness", "surprise"]

    for i in range(0, 1):
        correct = run_recognizer()
        print("got", correct, "percent correct!")

    fishface.save('models/emotion_detection_model.xml')
