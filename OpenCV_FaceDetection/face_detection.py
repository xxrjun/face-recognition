import cv2


def face_detect(img_path, fx, fy, sF, minN):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx, fy)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('face_detect.xml')
    faceRect = faceCascade.detectMultiScale(gray, scaleFactor=sF, minNeighbors=minN)

    for (x, y, w, h) in faceRect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return img


# detect
img1_path = 'src/moomoo.jpg'
img1 = face_detect(img1_path, 0.22, 0.22, 1.1, 3)

img2_path = 'src/people.jpg'
img2 = face_detect(img1_path, 0.6, 0.6, 1.1, 5)


img3_path = 'src/ghostintheshell2.jpg'
img3 = face_detect(img1_path, 0.2, 0.2, 1.1, 6)


cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey(0)

