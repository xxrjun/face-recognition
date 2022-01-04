import sqlite3
import io
import os
import numpy as np

import onnxruntime as ort
import cv2
from retinaface import RetinaFace
from skimage import transform as trans
from sklearn.preprocessing import normalize


# 1. face detection
def face_detect(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections


# 2. face alignment
def face_align(img_rgb, face_landmarks):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)  # 標準臉的 landmark points

    dst = np.array(face_landmarks, dtype=np.float32).reshape(5, 2)  # 將取得的 face_landmarks 作轉置
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    aligned_rgb = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)  # 仿射變換 affine transformation

    return aligned_rgb


# 3. feature extraction. return positions, landmarks, embeddings
def feature_extraction(img_rgb, detections):
    position = []
    landmarks = []
    embeddings = np.zeros([len(detections), 512])
    for i, face_info in enumerate(detections):
        face_position = [face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']]
        face_landmark = [face_info['left_eye'], face_info['right_eye'],
                         face_info['nose'], face_info['left_lip'], face_info['right_lip']]

        position.append(face_position)
        landmarks.append(face_landmark)

        aligned = face_align(img_rgb, landmarks)
        t_aligned = np.transpose(aligned, (2, 0, 1))

        inputs = t_aligned.astype(np.float32)
        input_blob = np.expand_dims(inputs, axis=0)

        first_input_name = sess.get_inputs()[0].name
        first_output_name = sess.get_outputs()[0].name

        prediction = sess.run([first_output_name], {first_input_name: input_blob})[0]
        final_embedding = normalize(prediction).flatten()

        embeddings[0] = final_embedding

    return position, landmarks, embeddings


detector = RetinaFace(quality='normal')
onnx_path = 'model/arcface_r100_v1.onnx'
EP_List = ['CPUExecutionProvider']
sess = ort.InferenceSession(onnx_path, providers=EP_List)

# 4. Create Database
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)  # Only needed here to simulate closing & reopening file
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# 將 file_path 底下的資料全部存進 file_data (list)
def load_file(file_path):
    file_Ddata = {}
    for person_name in os.listdir(file_path):
        person_file = os.path.join(file_path, person_name)

        total_pictures = []
        for picture in os.listdir(person_file):
            picture_path = os.path.join(person_file, picture)
            total_pictures.append(picture_path)

        file_data[person_name] = total_pictures

    return file_data


# converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# converts TEXT to np.array when selecting
sqlite3.register_converter("ARRAY", convert_array)

# 連接到 SQLite 數據庫。 若文件不存在則會自動創建
conn_db = sqlite3.connect('database.db')

# 創建表
conn_db.execute('CREATE TABLE face_info \
            (ID INT PRIMARY KEY NOT NULL, \
            NAME TEXT NOT NULL, \
            Embedding ARRAY NOT NULL)')

# 將 database 載入數據庫
file_path = 'database'
if os.path.exists(file_path):
    file_data = load_file(file_path)

    for i, person_name in enumerate(file_data.keys()):
        picture_path = file_data[person_name]
        sum_embeddings = np.zeros([1, 512])

        # 將所有同對象的圖片的臉部特徵值加總
        for j, picture in enumerate(picture_path):
            img_rgb, detections = face_detect(picture)
            position, landmarks, embeddings = feature_extraction(img_rgb, detections)
            sum_embeddings += embeddings

        final_embedding = sum_embeddings / len(picture_path)  # 平均值
        adapt_embedding = adapt_array(final_embedding)

        # 插入值
        conn_db.execute("INSERT INTO face_info (ID, NAME, Embeddings) VALUES (?, ?, ?)",
                        (i, person_name, adapt_embedding))
    conn_db.commit()
    conn_db.close()
















