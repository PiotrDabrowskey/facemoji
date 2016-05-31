import cv2

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def get_normalized_faces(image):
    faces_coordinates = find_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]

    return [normalize_face(face) for face in cutted_faces]

def normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face;

def find_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces  # list of (x, y, w, h)

if __name__ == "__main__":
    image = cv2.imread('test_data/test.jpg')
    cv2.imshow("face", image)
    index = 0
    for index, face in enumerate(get_normalized_faces(image)):
        cv2.imshow("face %s" %++index, face)

    cv2.waitKey(0)