import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

 #Define emotion order
participants = glob.glob("data/source_emotions/*") #Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s/*" %sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.

            images = glob.glob("data/source_images/%s/*" %(current_session))
            sourcefile_emotion = images[-1].split('/')[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = images[0].split('/')[-1] #do same for neutral image

            dest_neut = "data/sorted_set/neutral/%s" %sourcefile_neutral #Generate path to put neutral image
            dest_emot = "data/sorted_set/%s/%s" %(emotions[emotion], sourcefile_emotion) #Do same for emotion containing image

            copyfile("data/source_images/%s/%s" %(current_session, sourcefile_neutral), dest_neut) #Copy file
            copyfile("data/source_images/%s/%s" %(current_session, sourcefile_emotion), dest_emot) #Copy file