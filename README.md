### hey, what's that?

It's just a bunch of python scripts. Together they:

1. Harvest emotions dataset to extract faces from it in normalized way (same size, grey colours)

2. Teach a [fisherfaces classifier](http://www.scholarpedia.org/article/Fisherfaces) to classificate emotions

3. Swap faces to emoticons in real-time (using video stream from a webcam)

### how can I use it?

To see it in action you need:

1. Python

2. OpenCV

3. Dataset to teach a model, but you can used one provided by us. It was teached on http://www.consortium.ri.cmu.edu/ckagree/


### screenshot
![05:38](/facemoji_screenshot.png?raw=true "05:38")

### credits

We got the idea (and harvesting of files) to use Cohn-Kanade dataset to classificate emotions from [Paul van Gent's blog post](http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/), thanks for that!
