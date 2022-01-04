import cv2
from retinaface import RetinaFace

# init with normal accuracy option
detector = RetinaFace(quality='normal')

# 讀取圖檔，因為 opencv 預設是 bgr 因此要轉成 rgb
img_path = 'database/Elon Musk/elon_musk_1.jpg'
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 取得人臉及特徵的座標值 landmarks
detections = detector.predict(img_rgb)
print(detections)
print(len(detections))

# 將取得的座標值畫上記號及框框
img_result = detector.draw(img_rgb, detections)

# 轉回 cv2 可讀的 gbr
img = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

# show using cv2
cv2.imshow('img', img)
if cv2.waitKey(0) == ord('q'):  # press q to quit
    print('exit')

cv2.destroyWindow('img')

