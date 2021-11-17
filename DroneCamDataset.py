import os
import numpy as np
import cv2
import datetime
from djitellopy import tello
from pytz import timezone

print(1)


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

w, h = 360, 240
cap.set(3, w)
cap.set(4, h)

face_id = input('Input face ID : ')

me = tello.Tello()
me.connect()

me.streamon()

# 구분 ID 날짜로 바꿀 것
count = 0
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        cv2.rectangle (img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M"))
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + '.' + date + '.jpg', gray[y:y+h,x:x+w])
        print("dataset/User." + str(face_id) + '.' + str(count) + '.' + date + '.jpg', " was stored!")
        roi_gray = gray[y : y+h, x : x+w]
        roi_color = img[y : y+h, x : x+w]
    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    elif count >= 120:
        break

cap.release()
cv2.destroyAllWindows()

