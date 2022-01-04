import numpy as np
import sqlite3
import io
import os

import cv2
from retinaface import RetinaFace
from skimage import transform as trans
import onnxruntime as ort
from sklearn.preprocessing import normalize


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
        face_position = [face_info['x1'], face_info['y1'], face_info['x1'], face_info['y1']]
        face_landmark = [face_info['left_eye'], face_info['right_eye'],
                         face_info['nose'], face_info['left_lip'], face_info['right_lip']]

        positions.append(face_position)
        landmarks.append(face_landmark)

        aligned = face_align(img_rgb, landmarks)
        t_aligned = np.transpose(aligned, (2, 0, 1))

        inputs = t_aligned.astype(np.float32)
        input_blob = np.expand_dims(inputs, axis=0)

        first_input_name = sess.get_inputs()[0].name
        first_output_name = sess.get_outputs()[0].name

        prediction = sess.run([first_output_name], {first_input_name : input_blob})[0]
        final_embedding = normalize(prediction).flatten()

        embeddings[0] = final_embedding

    return positions, landmarks, embeddings


# ============================================
detector = RetinaFace(quality='normal')
onnx_path = 'model/arcface_r100_v1.onnx'
EP_List = ['CPUExecutionProvider']
sess = ort.InferenceSession(onnx_path, providers=EP_List)
img_path = 'src/people.jpg'
img_rgb, detections = face_detect(img_path)
position, landmarks, embeddings = feature_extract(img_rgb, detections)

# ============================================
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# 連接資料庫並取得內部所有資料
conn_db = sqlite3.connect('database.db')
cursor = conn_db.cursor()
db_data = cursor.fetchall()

# 跟 database 中的數據做比較
total_distances = []
total_names = []
for data in db_data:
    total_names.append(db_data[1])
    db_embeddings = convert_array(data[2])
    distance = round(np.linalg.norm(db_embeddings - embeddings), 2)
    total_distances.append(distance)

# 所有人比對的結果
total_result = dict(zip(total_names, total_distances))

# 找到距離最小者，也就是最像的人臉
idx_min = np.argmin(total_distances)

name, distance = total_names[idx_min], total_distances[idx_min]

# set threshold
threshold = 1

# 差異是否低於門檻
if distance < threshold:
    print('Found!', name, distance, total_result)
else:
    print('Unknown person', name, distance, total_result)




