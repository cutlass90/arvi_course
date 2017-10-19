import dlib
import cv2
import numpy as np
from scipy.spatial import distance
from keras.preprocessing import image
from scipy.ndimage import imread

shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords



def gen_landmark(image, path):
    # image - PIL image
    rects = detector(image, 1)
    nRects = len(rects)
    if nRects != 1:
        cv2.imwrite('failed.jpg', image)
        raise ValueError("Found {} faces on image in path {}".format(nRects, path))

    rect = rects[0]
    landmarks = predictor(image, rect)
    landmarks = shape_to_np(landmarks)
    normalized = normalize_landmarks(landmarks)
    return landmarks, normalized


def normalize_landmarks(landmarks):
    landmarks = landmarks.astype(np.float64)
    leftEye = np.mean(landmarks[36:42], axis=0).astype(np.int32)
    rightEye = np.mean(landmarks[42:48], axis=0).astype(np.int32)
    mPoint = (leftEye + rightEye) / 2
    landmarks -= mPoint
    d = distance.euclidean(leftEye, rightEye)
    landmarks /= d
    # landmarks -= meanLandmark  # todo what is that??
    return landmarks


if __name__ == '__main__':
    file_ = 'S.png'
    img = cv2.imread(file_)
    npshape, norm = gen_landmark(img)
    for (x, y) in npshape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite('landmarks.jpg', img)
