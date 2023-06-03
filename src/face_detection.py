import cv2


def face_detect(img_path, detector):
    """
    Detect faces in image.

    :param img_path:
    :type img_path: str
    :param detector:
    :type detector: RetinaFace
    :return: img_rgb, detections
    """

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections


def face_detect_bgr(img_bgr, detector):
    """
    Detect faces in image.

    :param img_bgr:
    :type img_bgr: numpy.ndarray
    :param detector:
    :type detector: RetinaFace
    :return: img_rgb, detections
    """

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections
