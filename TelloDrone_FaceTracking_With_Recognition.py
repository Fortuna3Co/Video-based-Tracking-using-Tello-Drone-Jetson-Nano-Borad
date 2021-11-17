import cv2
import numpy as np
from djitellopy import tello
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

id = 0
names = ['None', 'KHS', 'KSY', 'LDS']
font = cv2.FONT_HERSHEY_SIMPLEX

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
me.takeoff()

w, h = 360, 240
fbRange = [5000, 6000]
pid = [0.4, 0.4, 0]
px_Error, py_Error = 0, 0
no_face_count = 0


def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8, minSize = (int(360 * 0.05), int(240 * 0.05)))
    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
        
        id, confidence = recognizer.predict(imgGray[y : y+h, x : x+w])
        
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
    
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, px_Error, py_Error, no_face_count):
    area = info[1]
    x, y = info[0]
    fb = 0

    x_error = x - w // 2
    y_error = y - h // 2

    x_speed = pid[0] * x_error + pid[1] * (x_error - px_Error)
    x_speed = int(np.clip(x_speed, -100, 100))

    y_speed = pid[0] * y_error + pid[1] * (y_error - py_Error)
    y_speed = int(np.clip(y_speed, -100, 100))

    print("area : ", area)
    
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb += 20
    
    if  x == 0:
        no_face_count = 0
        x_speed = 0
        x_error = 0
        y_speed = 0
        y_error = 0
    
    
    me.send_rc_control(0, fb, -y_speed, x_speed)
    return x_error, y_error, no_face_count



#cap = cv2.VideoCapture(0)

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)
    px_Error, py_Error, no_face_count = trackFace(info, w, pid, px_Error, py_Error, no_face_count)
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("land")
        me.land()
        break

print("\n [INFO] Exiting Program and cleanup stuff")

img.release()
cv2.destroyAllWindows()



# 1. pyError 부분 수정 필요
# 2. cx, cy 일정 범위 안에만 들어오면 되게 
# 3. no_face_count 구현
