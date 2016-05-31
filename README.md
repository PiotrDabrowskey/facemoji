### hey, what's that?

It's just a bunch of python scripts. Together they:

1. Harvest emotions dataset to extract faces from it in normalized way (same size, grey colours)
2. Teach a [fisherfaces classifier](http://www.scholarpedia.org/article/Fisherfaces) to classificate emotions
3. Swap faces to emoticons in real-time (using video stream from a webcam)

### what steps to follow to run it?

If you don't have a dataset you can use a model teached by us and start from step 4:

1. Put CK emotions dataset inside /data/ folder following tips from [Paul van Gent's blog post](http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/)
2. Run process_dataset.py. It harvests dataset and puts neutral and emotions images into /data/sorted_set/, it also normalizes them
3. Run prepare_model.py to teach a model using /data/sorted_set/ files. You can specify list emotions you want to use. It saves a teached model to /models/emotion_detection_model.xml
4. Run webcam.py. It opens a webcam stream, detect emotions on faces (using /models/emotion_detection_model.xml) and changes them to specified emojis (/graphics/)

### what do i need to run it?

To see it in action you need:

1. Python
2. OpenCV
3. Dataset to teach a model, but you can used one provided by us. It was teached on http://www.consortium.ri.cmu.edu/ckagree/

### faq
Q: **I have my own emotion dataset. How can I use it with these scripts?**

A: You need to supply normalized images of faces with emotions. Use find_faces method from face_detect.py to find faces on your images. It returns cropped and normalized faces, save them to  to /data/sorted_set/%emotion_name%. Then run step 3 to teach a model and step 4 to begin swapping faces from webcam.


Q: **I want to use different emojis, for example my university profesors**

A: Place your university profesors heads inside /graphics/ folder following filenames convetion (filename should be an emotion label)

### screenshot
![05:38](/facemoji_screenshot.png?raw=true "05:38")

### credits

We got the idea (and harvesting of files) to use Cohn-Kanade dataset to classificate emotions from [Paul van Gent's blog post](http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/), thanks for that!
