import cv2
if cv2.__version__ == '3.1.0':
    from PIL import Image
else:
    import Image
from cv2 import WINDOW_NORMAL

from face_detect import detect_faces
from image_converters import nparray_as_image, image_as_nparray


def load_emoticons(emotions):
    emoticons = []
    for emotion in emotions:
        emoticons.append(nparray_as_image(cv2.imread('graphics/%s.png' %emotion, -1), mode=None))

    return emoticons

def show_webcam_and_run(model, emoticons):
    cv2.namedWindow('webcam', WINDOW_NORMAL)
    cv2.resizeWindow('webcam', 1600, 1200)
    # cv2.namedWindow('face_preview')

    vc = cv2.VideoCapture(0)
    rval, frame = vc.read() if vc.isOpened() else None

    while rval:
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            cropped = frame[y:y + h, x:x + w]
            if cropped is not None:  # if face detected
                image = cv2.resize(cropped, (350, 350))  # reize to model's input
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                # cv2.imshow('face_preview', gray)  # face visualization

                prediction = model.predict(gray)  # do prediction
                img_to_draw = emoticons[prediction[0]]
                img_to_draw = img_to_draw.resize((h, w), Image.ANTIALIAS)
                emotion = image_as_nparray(img_to_draw)  # convert to nparray
                for c in range(0, 3):
                    frame[y:y + h, x:x + w, c] = emotion[:, :, c] * (emotion[:, :, 3] / 255.0) \
                                                 + frame[y:y + h, x:x + w, c] * (1.0 - emotion[:, :, 3] / 255.0)

        cv2.imshow('webcam', frame)

        rval, frame = vc.read()
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow('webcam')
    cv2.destroyWindow('face_preview')


if __name__ == '__main__':
    emotions = ["neutral", "anger", "disgust", "happy", "sadness", "surprise"]
    emoticons = load_emoticons(emotions)

    # load model
    if cv2.__version__ == '3.1.0':
        fisher_face = cv2.face.createFisherFaceRecognizer()
    else:
        fisher_face = cv2.createFisherFaceRecognizer()
    fisher_face.load('models/emotion_detection_model.xml')

    # use learnt model
    show_webcam_and_run(fisher_face, emoticons)
