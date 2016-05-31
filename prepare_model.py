import glob
import random
import numpy as np
import cv2

from image_commons import load_image

fishface = cv2.createFisherFaceRecognizer()
training_set_size = 0.95

def get_files(emotion):
    files = glob.glob("data/sorted_set/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]
    return training, prediction

def make_sets():
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
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    correct = sum(1 for id, image in enumerate(prediction_data) if fishface.predict(image)[0] == prediction_labels[id])

    return ((100 * correct) / len(prediction_data))

if __name__ == '__main__':
    emotions = ["neutral", "anger", "disgust", "happy", "sadness", "surprise"]

    for i in range(0, 2):
        correct = run_recognizer()
        print "got", correct, "percent correct!"

    fishface.save('models/emotion_detection_model.xml')
