import numpy as np
import cv2
from retinaface import RetinaFace
from skimage import transform as trans


# face detection
def face_detect(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections


detector = RetinaFace(quality='normal')  # init with normal accuracy option
img_path = 'database/Elon Musk/elon_musk_1.jpg'
img_rgb, detections = face_detect(img_path)
img_result = detector.draw(img_rgb, detections)
img = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

# 標準臉的 landmark points。許多網站都以此二維陣列作為標準臉關鍵點
src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)

# 取得臉部特徵點座標 face_landmark points
face_landmarks = []
for i, face_info in enumerate(detections):
    face_landmarks = [face_info['left_eye'], face_info['right_eye'], face_info['nose'], face_info['left_lip'],
                      face_info['right_lip']]

# 將取得的 face_landmarks 作轉置 => 跟 src 矩陣一樣
dst = np.array(face_landmarks, dtype=np.float32).reshape(5, 2)

# 把 face_landmarks 跟 標準臉的 landmark points對齊
tform = trans.SimilarityTransform()  # 轉換矩陣 transformation matrix
tform.estimate(dst, src)  # 從一組對應點估計變換矩陣。 return True, if model estimation succeeds.
M = tform.params[0:2, :]  # 要用的轉換矩陣

# 仿射變換 affine transformation : cv2.warpAffine(輸入圖檔, 轉換矩陣, 輸出圖像大小, 邊界填充值)
aligned_rgb = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)
aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)  # 轉回 bgr 才能用 opencv show 出來q

# show using cv2
cv2.imshow('img', img)
cv2.imshow('aligned_bgr', aligned_bgr)
if cv2.waitKey(0) == ord('q'):  # press q to quit
    print('exit')

cv2.destroyWindow('aligned')
cv2.destroyWindow('img')


