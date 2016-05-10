import Image

import cv2

from face_detect import detect_face_and_crop
from image_converters import nparray_as_image, image_as_nparray


def show_webcam_and_detect_face():
    cv2.namedWindow('webcam')
    cv2.namedWindow('face_preview')
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        frame = None

    while rval:
        cv2.imshow('webcam', frame)

        face_from_frame = detect_face_and_crop(frame)
        if face_from_frame is not None:
            image = nparray_as_image(face_from_frame)
            image = image.resize((200, 200), Image.ANTIALIAS)

            face_from_frame = image_as_nparray(image)
            cv2.imshow('face_preview', face_from_frame)

        rval, frame = vc.read()
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow('webcam')
    cv2.destroyWindow('face_preview')


if __name__ == '__main__':
    # load model
    fisher_face = cv2.createFisherFaceRecognizer()
    fisher_face.load('models/emotion_detection_model.xml')

    # use learnt model
    show_webcam_and_detect_face()
