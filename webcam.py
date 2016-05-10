import Image

import cv2

from face_detect import detect_faces, crop_face
from image_converters import nparray_as_image, image_as_nparray


def show_webcam_and_run(model):
    cv2.namedWindow('webcam')
    cv2.namedWindow('face_preview')

    happy_img = nparray_as_image(cv2.imread('graphics/happy.png', -1), mode=None)
    sad_img = nparray_as_image(cv2.imread('graphics/sad.png', -1), mode=None)

    vc = cv2.VideoCapture(0)
    rval, frame = vc.read() if vc.isOpened() else None  # try to get the first frame



    while rval:
        faces = detect_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cropped = frame[y:y + h, x:x + w]
            if cropped is not None:  # if face detected
                image = nparray_as_image(cropped)  # convert to image
                cropped = image_as_nparray(image)  # convert back to nparray
                image = cv2.resize(cropped, (350, 350))  # reize to model's input
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                prediction = model.predict(gray)  # do prediction
                cv2.imshow('face_preview', gray)  # face visualization
                print(prediction[0])
                if prediction[0] > 1:
                    img_to_draw = happy_img if prediction[0] == 2 else sad_img  # select emotion's image
                    img_to_draw = img_to_draw.resize((h, w), Image.ANTIALIAS)  # resize to face's size
                    emotion = image_as_nparray(img_to_draw)  # convert to nparray
                    for c in range(0, 3):
                        frame[y:y + h, x:x + w, c] = emotion[:, :, c] * (emotion[:, :, 3] / 255.0) \
                                                     + frame[y:y + h, x:x + w, c] * (1.0 - emotion[:, :, 3] / 255.0)

        cv2.imshow('webcam', frame)

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
    show_webcam_and_run(fisher_face)
