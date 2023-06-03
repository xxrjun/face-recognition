import numpy as np
from face_alignment import face_align
from sklearn.preprocessing import normalize


def feature_extract(img_rgb, detections, sess):
    """
    Extracts the features from the detected faces

    :param img_rgb: The input RGB image where faces are detected
    :type img_rgb: np.ndarray
    :param detections: The list of detected face information. Each detection includes the positions and
                        landmarks of a detected face.
    :type detections: list
    :param sess: The tensorflow session used to run the model for feature extraction.
    :type sess: tf.Session
    :return: A tuple of lists containing the face positions, landmarks, and embeddings respectively
                for each detected face.
    """

    positions = []  # To store face positions for all detections
    landmarks = []  # To store face landmarks for all detections
    embeddings = np.zeros((len(detections), 512))  # To store face embeddings for all detections

    for i, face_info in enumerate(detections):  # Loop through each detected face
        # Get the position and landmarks from detection information
        face_position = [face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']]
        face_landmarks = [face_info['left_eye'], face_info['right_eye'], face_info['nose'], face_info['left_lip'],
                          face_info['right_lip']]

        positions.append(face_position)  # Append the position to the list
        landmarks.append(face_landmarks)  # Append the landmarks to the list

        # Align the face based on landmarks and transpose the result
        aligned = face_align(img_rgb, face_landmarks)
        t_aligned = np.transpose(aligned, (2, 0, 1))

        # Prepare the inputs for the model
        inputs = t_aligned.astype(np.float32)
        input_blob = np.expand_dims(inputs, axis=0)

        # Get the input and output names of the model
        first_input_name = sess.get_inputs()[0].name
        first_output_name = sess.get_outputs()[0].name

        # Run the model to get the embeddings
        prediction = sess.run([first_output_name], {first_input_name: input_blob})[0]
        final_embedding = normalize(prediction).flatten()  # Flatten and normalize the embeddings

        embeddings[i] = final_embedding  # Store the embeddings

    # Return the positions, landmarks and embeddings
    return positions, landmarks, embeddings
