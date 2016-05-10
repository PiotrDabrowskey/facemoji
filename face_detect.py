import cv2

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces  # list of (x, y, w, h)


def crop_face(image, x, y, h, w):
    if w == 0 or h == 0 or y + h > image.shape[0] or x + w > image.shape[1]:
        return None
    return image[y:y + h, x:x + w]


if __name__ == "__main__":
    image = cv2.imread('data/happy/139,131,192,128.jpg')
    faces = detect_faces(image)
    (x, y, w, h) = faces[0]
    cropped = crop_face(image, x, y, w, h)
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
