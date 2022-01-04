import onnxruntime as ort
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from retinaface import RetinaFace
from skimage import transform as trans


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


detector = RetinaFace(quality='normal')  # init with normal accuracy option
img_path = 'database/Elon Musk/elon_musk_1.jpg'
img_rgb, detections = face_detect(img_path)

# 取得臉部位置 positions 及特徵點座標 landmark points
face_positions = []
face_landmarks = []
for i, face_info in enumerate(detections):
    face_positions = [face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']]
    face_landmarks = [face_info['left_eye'], face_info['right_eye'], face_info['nose'], face_info['left_lip'],
                      face_info['right_lip']]


# 3. Feature Extraction
onnx_path = 'model/arcface_r100_v1.onnx'
EP_list = ['CPUExecutionProvider']

# Create inference session
sess = ort.InferenceSession(onnx_path, providers=EP_list)

aligned = face_align(img_rgb, face_landmarks)  # 取得對齊後的圖片
t_aligned = np.transpose(aligned, (2, 0, 1))  # 轉置

inputs = t_aligned.astype(np.float32)  # 將轉置後的人臉轉換 dtype 為 float32
input_blob = np.expand_dims(inputs, axis=0)  # 擴充矩陣維度，因為後續函式需要 二維矩陣

first_input_name = sess.get_inputs()[0].name  # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
first_output_name = sess.get_outputs()[0].name  # get the inputs metadata as a list of :class:`onnxruntime.NodeArg`

# inference run using image_data as the input to the model
# pass a tuple rather than a single numpy ndarray.
prediction = sess.run([first_output_name], {first_input_name: input_blob})[0]

# 進行正規化並且轉成一維陣列
final_embedding = normalize(prediction).flatten()
print(final_embedding)
