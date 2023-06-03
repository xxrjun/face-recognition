import cv2
import numpy as np
from skimage import transform as trans


def face_align(img_rgb, landmarks):
    """
    This function aligns the face based on the 5 facial landmarks.

    :param img_rgb:
    :type img_rgb: np.ndarray
    :param landmarks:
    :type landmarks: list
    :return:
    """

    # These are predefined coordinates for 5 facial landmarks based on the standard 112x112 image.
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    # Convert landmarks to numpy array and reshape it to (5, 2) for further operations.
    dst = np.array(landmarks, dtype=np.float32).reshape(5, 2)

    # Use similarity transform from skimage library to estimate the transformation matrix.
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)

    # Extract the transformation matrix from the transformation object.
    M = tform.params[0:2, :]

    # Apply the transformation matrix to the image to align the face.
    # Here, borderValue=0 is used to fill pixels outside the input boundaries.
    aligned = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)

    return aligned
