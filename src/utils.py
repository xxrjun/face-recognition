import os
import numpy as np
import sqlite3
from database import create_db, adapt_array, convert_array
from retinaface import RetinaFace
import onnxruntime as ort

# initial detector and model
detector = RetinaFace(quality='normal')
onnx_path = '../model/arcface_r100_v1.onnx'
sess = ort.InferenceSession(onnx_path)

# create db
db_path = '../database/database.db'
file_path = '../database'
sqlite3.register_adapter(np.array, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

if not os.path.exists(db_path):
    create_db(db_path, file_path, detector, sess)
