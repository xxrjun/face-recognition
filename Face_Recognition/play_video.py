import cv2



# 取得視訊鏡頭
cap = cv2.VideoCapture('src/ericWang.mp4')
while True:  # 讀取每一幀去播放影片
    ret, frame = cap.read()  # 回傳兩個值(boolean, next frame)
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        cv2.imshow('video', frame)
    else:
        break
    if cv2.waitKey(1) == ord('q'):  # 如果按下 q 就會跳出
        break

cap.release()
cv2.destroyAllWindows()