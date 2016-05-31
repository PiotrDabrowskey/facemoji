"""
This file is responsible for harvesting CK database for images of emotions. It gets a neutral face and a emotion face for each subject.
Based on Paul van Gent's code from blog post: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
"""
import glob
import os
from shutil import copyfile

import cv2

from face_detect import find_faces


def remove_old_set(emotions):
    """
    Removes old images produced from dataset.
    :param emotions: List of emotions names.
    """
    print("Removing old dataset")
    for emotion in emotions:
        filelist = glob.glob("data/sorted_set/%s/*" % emotion)
        for f in filelist:
            os.remove(f)


def harvest_dataset(emotions):
    """
    Copies photos of participants in sessions to new folder.
    :param emotions: List of emotions names.
    """
    print("Harvesting dataset")
    participants = glob.glob('data/source_emotions/*')  # returns a list of all folders with participant numbers

    for participant in participants:
        neutral_added = False

        for sessions in glob.glob("%s/*" % participant):  # store list of sessions for current participant
            for files in glob.glob("%s/*" % sessions):
                current_session = files[20:-30]
                file = open(files, 'r')

                # emotions are encoded as a float, readline as float, then convert to integer
                emotion = int(float(file.readline()))
                images = glob.glob("data/source_images/%s/*" % current_session)

                # get path for last image in sequence, which contains the emotion
                source_filename = images[-1].split('/')[-1]
                # do same for emotion containing image
                destination_filename = "data/sorted_set/%s/%s" % (emotions[emotion], source_filename)
                # copy file
                copyfile("data/source_images/%s/%s" % (current_session, source_filename), destination_filename)

                if not neutral_added:
                    # do same for neutral image
                    source_filename = images[0].split('/')[-1]
                    # generate path to put neutral image
                    destination_filename = "data/sorted_set/neutral/%s" % source_filename
                    # copy file
                    copyfile("data/source_images/%s/%s" % (current_session, source_filename), destination_filename)
                    neutral_added = True


def extract_faces(emotions):
    """
    Crops faces in emotions images.
    :param emotions: List of emotions names.
    """
    print("Extracting faces")
    for emotion in emotions:
        photos = glob.glob('data/sorted_set/%s/*' % emotion)

        for file_number, photo in enumerate(photos):
            frame = cv2.imread(photo)
            normalized_faces = find_faces(frame)
            os.remove(photo)

            for face in normalized_faces:
                try:
                    cv2.imwrite("data/sorted_set/%s/%s.png" % (emotion, file_number + 1), face[0])  # write image
                except:
                    print("error in processing %s" % photo)


if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    remove_old_set(emotions)
    harvest_dataset(emotions)
    extract_faces(emotions)
