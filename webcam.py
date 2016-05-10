import cv2

from face_detect import detect_face_and_crop


def show_webcam_and_detect_face():
    cv2.namedWindow("preview")
    cv2.namedWindow("face")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        face_from_frame = detect_face_and_crop(frame)
        if face_from_frame is not None:
            cv2.imshow('face', face_from_frame)
        rval, frame = vc.read()
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")
    cv2.destroyWindow("face")


if __name__ == "__main__":
    # learn

    # use learnt model
    show_webcam_and_detect_face()
