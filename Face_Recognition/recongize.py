from retinaface import RetinaFace
import cv2
import numpy as np
from skimage import transform as trans
import onnxruntime as ort
from sklearn.preprocessing import normalize
import io
import os
import sqlite3
import matplotlib.pyplot as plt


# 1. face detection
def face_detect(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections


# 2. face alignment
def face_align(img_rgb, landmarks):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    dst = np.array(landmarks, dtype=np.float32).reshape(5, 2)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)

    return aligned


# 3. feature extraction
def feature_extract(img_rgb, detections):
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


# 4. Data base
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def load_file(file_path):
    file_data = {}
    for person_name in os.listdir(file_path):
        person_dir = os.path.join(file_path, person_name)

        person_pictures = []
        for picture in os.listdir(person_dir):
            picture_path = os.path.join(person_dir, picture)
            person_pictures.append(picture_path)

        file_data[person_name] = person_pictures

    return file_data


# 4. Create Database
def create_db(db_path, file_path):
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
                img_rgb, detections = face_detect(picture)
                position, landmarks, embeddings = feature_extract(img_rgb, detections)
                sum_embeddings += embeddings

            final_embedding = sum_embeddings / len(picture_path)
            adapt_embedding = adapt_array(final_embedding)

            conn_db.execute("INSERT INTO face_info (ID, NAME, Embeddings) VALUES (?, ?, ?)",
                            (i, person_name, adapt_embedding))
        conn_db.commit()
        conn_db.close()
    else:
        print('database path does not exist')


# 5. face recognition
def compare_face(embeddings, threshold):
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


# ============================================
# initial detector and model
detector = RetinaFace(quality='normal')
onnx_path = 'model/arcface_r100_v1.onnx'
EP_List = ['CPUExecutionProvider']
sess = ort.InferenceSession(onnx_path, providers=EP_List)

# create db
db_path = 'database.db'
file_path = 'database'
sqlite3.register_adapter(np.array, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

if not os.path.exists(db_path):
    create_db(db_path, file_path)

# load picture we want to recognize
img_path = 'src/four_people.jpg'
img_rgb, detections = face_detect(img_path)
position, landmarks, embeddings = feature_extract(img_rgb, detections)

# set threshold and compare every face in the picture
threshold = 1
for i, embedding in enumerate(embeddings):
    name, distance, total_result = compare_face(embedding, threshold)
    print('total_result', total_result)

    cv2.rectangle(img_rgb, (position[i][0], position[i][1]), (position[i][2], position[i][3]), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_rgb, name + ', ' + str(distance), (position[i][0] + 10, position[i][1] - 10), font, 0.8,
                (0, 255, 0), 2)

# show the result
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb / 255)
_ = plt.axis('off')
plt.show()
