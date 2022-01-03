import cv2

cap = cv2.VideoCapture('src/色鬼.mp4')
# print(cap.get(cv2.CAP_PROP_FPS))
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier('face_detect.xml')
        faceRect = faceCascade.detectMultiScale(gray, 1.1, 8)
        for (x, y, w, h) in faceRect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.GaussianBlur(frame, (9, 9), 20, faceRect)
        cv2.imshow('video', frame)
    else:
        break

    if cv2.waitKey(1) == ord('q'):  # q = quit
        break