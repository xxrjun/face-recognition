import cv2

# 辨識圖片
img1 = cv2.imread('src/moomoo.jpg')
img1 = cv2.resize(img1, (0, 0), fx=0.22, fy=0.22)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier('face_detect.xml')
faceRect = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(len(faceRect))

for (x, y, w, h) in faceRect:
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 4)


img2 = cv2.imread('src/people.jpg')
img2 = cv2.resize(img2, (0, 0), fx=0.6, fy=0.6)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier('face_detect.xml')
faceRect = faceCascade.detectMultiScale(gray, 1.1, 5)
print(len(faceRect))

for (x, y, w, h) in faceRect:
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 4)


img3 = cv2.imread('src/ghostintheshell2.jpg')
img3 = cv2.resize(img3, (0, 0), fx=0.2, fy=0.2)
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier('face_detect.xml')
faceRect = faceCascade.detectMultiScale(gray, 1.1, 6)
print(len(faceRect))

for (x, y, w, h) in faceRect:
    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 4)
blurImg3 = cv2.GaussianBlur(img3, (3, 3), 10000, faceRect)


cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey(0)

