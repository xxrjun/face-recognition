import numpy as np
from face_alignment import face_align
from sklearn.preprocessing import normalize


def feature_extract(img_rgb, detections, sess):
    """
    Extracts the features from the detected faces

    :param img_rgb:
    :type img_rgb: np.ndarray
    :param detections:
    :type detections: list
    :param sess:
    :type sess: tf.Session
    :return: positions, landmarks, embeddings
    """

    positions = []
    landmarks = []
    embeddings = np.zeros((len(detections), 512))
    for i, face_info in enumerate(detections):
        face_position = [face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']]
        face_landmarks = [face_info['left_eye'], face_info['right_eye'],
                          face_info['nose'], face_info['left_lip'], face_info['right_lip']]

        positions.append(face_position)
        landmarks.append(face_landmarks)

        aligned = face_align(img_rgb, face_landmarks)
        t_aligned = np.transpose(aligned, (2, 0, 1))

        inputs = t_aligned.astype(np.float32)
        input_blob = np.expand_dims(inputs, axis=0)

        first_input_name = sess.get_inputs()[0].name
        first_output_name = sess.get_outputs()[0].name

        prediction = sess.run([first_output_name], {first_input_name: input_blob})[0]
        final_embedding = normalize(prediction).flatten()

        embeddings[i] = final_embedding

    return positions, landmarks, embeddings
