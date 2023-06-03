import sqlite3
import numpy as np
from database import convert_array


def compare_face(embeddings, threshold, db_path):
    """
    Compare face embeddings with database.

    :param embeddings: The embeddings of the detected face to be compared.
    :type embeddings: numpy array
    :param threshold: The threshold for comparison. If the distance between the embeddings is larger than the threshold,
                      the face is considered unknown.
    :type threshold: float
    :param db_path: The path to the database where known face information is stored.
    :type db_path: str
    :return: The name of the matched person, the minimum distance, and a dictionary of distances to all persons in the database.
    :rtype: tuple
    """

    # Connect to the database
    conn_db = sqlite3.connect(db_path)

    # Get all the data from the database
    cursor = conn_db.execute("SELECT * FROM face_info")
    db_data = cursor.fetchall()

    total_distances = []  # To store distances to all persons in the database
    total_names = []  # To store the names of all persons in the database
    for data in db_data:
        total_names.append(data[1])  # Append the name
        db_embeddings = convert_array(data[2])  # Convert the stored array back to numpy array

        # Compute the Euclidean distance between the detected face embeddings and the stored embeddings
        distance = round(np.linalg.norm(db_embeddings - embeddings), 2)
        total_distances.append(distance)  # Append the distance

    # Create a dictionary with names and distances
    total_result = dict(zip(total_names, total_distances))

    # Find the index of the minimum distance
    idx_min = np.argmin(total_distances)

    # Get the name and distance of the closest match
    name, distance = total_names[idx_min], total_distances[idx_min]

    # If the distance is larger than the threshold, consider the face as unknown
    if distance > threshold:
        name = 'Unknown person'

    # Return the name, distance, and the dictionary of results
    return name, distance, total_result

