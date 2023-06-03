import sqlite3
import numpy as np
from database import convert_array


def compare_face(embeddings, threshold, db_path):
    """
    Compare face embeddings with database.

    :param embeddings:
    :type embeddings: numpy array
    :param threshold:
    :type threshold: float
    :param db_path
    :type db_path: str
    :return: name, distance, total_result
    """

    conn_db = sqlite3.connect(db_path)
    cursor = conn_db.execute("SELECT * FROM face_info")
    db_data = cursor.fetchall()

    total_distances = []
    total_names = []
    for data in db_data:
        total_names.append(data[1])
        db_embeddings = convert_array(data[2])
        distance = round(np.linalg.norm(db_embeddings - embeddings), 2)
        total_distances.append(distance)

    total_result = dict(zip(total_names, total_distances))
    idx_min = np.argmin(total_distances)

    name, distance = total_names[idx_min], total_distances[idx_min]

    if distance > threshold:
        name = 'Unknown person'

    return name, distance, total_result
