import sqlite3
import os
import numpy as np
import io
from face_detection import face_detect
from feature_extraction import feature_extract


def adapt_array(arr):
    """
    Convert numpy array to sqlite Binary data.

    :param arr: numpy array
    :return: sqlite Binary data
    """

    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)

    return sqlite3.Binary(out.read())


def convert_array(text):
    """
    Convert sqlite Binary data to numpy array.

    :param text: sqlite Binary data
    :type text: sqlite Binary data
    :return: numpy array
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def load_file(file_path):
    """
    Load images from file path.

    :param file_path:
    :type file_path: str
    :return: file_data
    """

    file_data = {}
    for person_name in os.listdir(file_path):
        person_dir = os.path.join(file_path, person_name)

        person_pictures = []
        for picture in os.listdir(person_dir):
            picture_path = os.path.join(person_dir, picture)
            person_pictures.append(picture_path)

        file_data[person_name] = person_pictures

    return file_data


def create_db(db_path, file_path, detector, sess):
    """
    Create database from file path.

    :param db_path:
    :type db_path: str
    :param file_path:
    :type file_path: str
    :param detector:
    :type detector: dlib.fhog_object_detector
    :param sess:
    :type sess: tf.Session
    :return:
    """

    if os.path.exists(file_path):
        conn_db = sqlite3.connect(db_path)
        conn_db.execute("CREATE TABLE face_info \
                            (ID INT PRIMARY KEY NOT NULL, \
                             NAME TEXT NOT NULL, \
                            Embeddings ARRAY NOT NULL)")
        file_data = load_file(file_path)
        for i, person_name in enumerate(file_data.keys()):
            picture_path = file_data[person_name]
            sum_embeddings = np.zeros([1, 512])
            for j, picture in enumerate(picture_path):
                img_rgb, detections = face_detect(picture, detector)
                position, landmarks, embeddings = feature_extract(img_rgb, detections, sess)
                sum_embeddings += embeddings

            final_embedding = sum_embeddings / len(picture_path)
            adapt_embedding = adapt_array(final_embedding)

            conn_db.execute("INSERT INTO face_info (ID, NAME, Embeddings) VALUES (?, ?, ?)",
                            (i, person_name, adapt_embedding))
        conn_db.commit()
        conn_db.close()
    else:
        print('database path does not exist')
