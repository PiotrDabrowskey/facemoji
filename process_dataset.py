import cv2, os
import glob
from face_detect import get_normalized_faces

from shutil import copyfile

def remove_old_set(emotions):
    print('Removing old dataset')
    for emotion in emotions:
        filelist = glob.glob("data/sorted_set/%s/*" %emotion)
        for f in filelist:
            os.remove(f)


def harvest_dataset(emotions):
    print('Harvesting dataset')
    participants = glob.glob("data/source_emotions/*")  # Returns a list of all folders with participant numbers

    for x in participants:
        neutralAdded = False;

        part = "%s" %x[-4:] #store current participant number
        for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
            for files in glob.glob("%s/*" %sessions):
                current_session = files[20:-30]
                file = open(files, 'r')

                emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
                images = glob.glob("data/source_images/%s/*" %(current_session))

                sourcefile_emotion = images[-1].split('/')[-1] #get path for last image in sequence, which contains the emotion
                dest_emot = "data/sorted_set/%s/%s" %(emotions[emotion], sourcefile_emotion) #Do same for emotion containing image
                copyfile("data/source_images/%s/%s" %(current_session, sourcefile_emotion), dest_emot) #Copy file

                if not neutralAdded:
                    sourcefile_neutral = images[0].split('/')[-1]  # do same for neutral image
                    dest_neut = "data/sorted_set/neutral/%s" % sourcefile_neutral  # Generate path to put neutral image
                    copyfile("data/source_images/%s/%s" % (current_session, sourcefile_neutral), dest_neut)  # Copy file
                    neutralAdded = True

def extract_faces(emotions):
    print('Extracting faces')
    for emotion in emotions:
        photos = glob.glob('data/sorted_set/%s/*' % emotion)
        filenumber = 1

        for photo in photos:
            frame = cv2.imread(photo)
            normalized_faces = get_normalized_faces(frame)
            os.remove(photo)

            for face in normalized_faces:
                try:
                    cv2.imwrite("data/sorted_set/%s/%s.png" % (emotion, filenumber), face)  # Write image
                except:
                    print('error in processing %s' %photo)

                filenumber += 1  # Increment image number

if __name__ == '__main__':
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    remove_old_set(emotions)
    harvest_dataset(emotions)
    extract_faces(emotions)