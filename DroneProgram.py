import cv2
import numpy as np
import datetime
import os
from time import sleep
from djitellopy import tello
from cvzone.HandTrackingModule import HandDetector
from pytz import timezone
from PIL import Image

######################################## Tello Drone Face Tracking & Recognition
def tello_init():
    w, h = 800, 600
    pxError, pyError = 0, 0
    fbRange = [10000, 11000]
    pid = [0.1, 0.1, 0]
    return w, h, pxError, pyError, fbRange, pid

def TelloDrone_Connect():
    me = tello.Tello()
    me.connect()
    print(me.get_battery())
    return me

def TelloDrone_takeOff(me):
    me.streamon()
    me.takeoff()

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
    
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace(info, w, h, pid, px_Error, py_Error, fbRange):
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
    
    
    if x == 0:
        x_speed = 0
        x_error = 0
        y_speed = 0
        y_error = 0
    
    
    me.send_rc_control(0, fb, -y_speed, x_speed)
    return x_error, y_error

def findFaceWithRecognition(img):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainer/trainer.yml') # set Recognizer
    
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
        
        id, confidence = face_recognizer.predict(imgGray[y : y+h, x : x+w])
        
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

def making_dataset():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    md_cap = cv2.VideoCapture(0)

    w, h = 360, 240
    md_cap.set(3, w)
    md_cap.set(4, h)

    face_id = input('Input face ID : ')

    # 구분 ID 날짜로 바꿀 것
    count = 1
    while True:
        _, md_img = md_cap.read()
        gray = cv2.cvtColor(md_img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

        for (x, y, w, h) in faces:
            cv2.rectangle (md_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M"))
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + '.' + date + '.jpg', gray[y:y+h,x:x+w])
            print("dataset/User." + str(face_id) + '.' + str(count) + '.' + date + '.jpg', " was stored!")
            roi_gray = gray[y : y+h, x : x+w]
            roi_color = md_img[y : y+h, x : x+w]
        cv2.imshow('video', md_img)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif count >= 30:
            break

    md_cap.release()
    cv2.destroyWindow('video')

def train_dataset():
    path = 'dataset'
    train_recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    def getImageAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePaths in imagePaths:
            PIL_img = Image.open(imagePaths).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePaths)[-1].split('.')[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y : y+h, x : x+w])
                ids.append(id)
        
        return faceSamples, ids

    faces, ids = getImageAndLabels(path)
    train_recognizer.train(faces, np.array(ids))

    train_recognizer.write('trainer/trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
#######################################

start_menu_select_flag = 0
faceTracking_menu_select_flag = 0
faceTrackingRecognition_menu_select_flag = 0
directControl_menu_select_flag = 0

webcam_w, webcam_h = 1280, 720  
detector = HandDetector(detectionCon=1)
start_menu_info, faceTracking_menu = [], []
faceTrackingRecognition_menu, directControl_menu = [], []
webCamDataset_menu = []

names = ['None', 'KHS', 'KSY', 'LDS'] # For Recognition ID
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read('trainer/trainer.yml') # set Recognizer
font = cv2.FONT_HERSHEY_SIMPLEX # print Recgniition ID


## Create class for display
class display_menu():
    def __init__(self, pos, text, size=[800, 40]):
        self.pos = pos
        self.text = text
        self.size = size

## Start Menu Info Class append
start_menu_info.append(display_menu([40, 50], '1. Face Tracking'))
start_menu_info.append(display_menu([40, 90], '2. Face Tracking with Recognition'))
start_menu_info.append(display_menu([40, 130], '3. Direct Control'))
start_menu_info.append(display_menu([40, 170], '4. WebCam Dataset'))
start_menu_info.append(display_menu([40, 210], '5. Quit'))
##

## Face Tracking Menu Class append
faceTracking_menu.append(display_menu([40, 650], 'Quit this mode'))
faceTracking_menu.append(display_menu([950, 50], 'Picture'))
faceTracking_menu.append(display_menu([950, 150], 'Tello Picture'))
##

## Face Tracking with Recognition Menu Class append
faceTrackingRecognition_menu.append(display_menu([40, 600], 'Quit this mode'))
faceTrackingRecognition_menu.append(display_menu([950, 50], 'Picture'))
faceTrackingRecognition_menu.append(display_menu([950, 150], 'Tello Picture'))
##

## Direct Control Menu Class append
directControl_menu.append(display_menu([50, 200], 'OFF/ON'))
directControl_menu.append(display_menu([50, 350], '  UP  '))
directControl_menu.append(display_menu([50, 450], ' DOWN '))
directControl_menu.append(display_menu([1000, 350], ' LEFT '))
directControl_menu.append(display_menu([1100, 350], ' RIGHT '))
directControl_menu.append(display_menu([1050, 250], '  FW  '))
directControl_menu.append(display_menu([1050, 450], '  BW  '))
directControl_menu.append(display_menu([40, 600], ' Quit '))

directControl_menu.append(display_menu([950, 50], 'Picture'))
directControl_menu.append(display_menu([890, 150], 'Tello Picture'))
##

## WebCam Dataset Menu Class append
webCamDataset_menu.append(display_menu([40, 90], '1. Collect Dataset'))
webCamDataset_menu.append(display_menu([40, 130], '2. Recognition'))
webCamDataset_menu.append(display_menu([40, 650], 'Quit this mode'))
##

# Initialize webcam camera with width & height
# return cv2.VideoCapture
def init_webcam_camera(w, h):
    cap = cv2.VideoCapture(0)
    cap.set(3, w)
    cap.set(4, h)
    return cap

# Initialize menu display
def init_menu_display(img):
    #cv2.rectangle(img, (20, 20), (1000, 60), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Welcome To Tello Drone AI Program!!", (20 + 20, 20 + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    return img

# display start menu
# When select the menu using hand(finger), each menu return a number(menu_select_falg)
def display_start_menu(img, start_menu_info, lmList, detector):
    
    for index, menu in enumerate(start_menu_info, 0):
        x, y = menu.pos
        w, h = menu.size
        
        cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        
        if lmList:
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                
                l, _, _, = detector.findDistance(8, 12, img, draw=False)
                print(l)
                
                if l < 50:
                    cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                    print('Clicked : ', index + 1)
                    return img, index + 1
    
    return img, 0

# display menus
def display_menu(img, menu_info, lmList, detector):
    
    for index, menu in enumerate(menu_info, 0):
        x, y = menu.pos
        w, h = menu.size
        cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        
        if lmList:
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    
                    l, _, _, = detector.findDistance(8, 12, img, draw=False)
                    
                    if l < 50:
                        cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        print('Clicked : ', index + 1)
                        
                        return img, index + 1

    return img, 0

# display direct control menus
def direct_control_display_menu(img, menu_info, lmList, detector):
    
    for index, menu in enumerate(menu_info, 0):
        x, y = menu.pos
        w, h = menu.size
        text_size = len(menu.text)
        
        cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.circle(img, (x + text_size * 10, y - 10), 50, (0, 0, 0))
        
        if lmList:
            if x < lmList[8][0] < x + text_size * 20 and y - 40 < lmList[8][1] < y + 40:
                    cv2.circle(img, (x + text_size * 10, y - 10), 50, (0, 255, 0), 8)
                    
                    l, _, _, = detector.findDistance(8, 12, img, draw=False)
                    
                    if l < 50:
                        cv2.circle(img, (x + text_size * 10, y - 10), 50, (0, 255, 255), 8)
                        print('Clicked : ', index + 1)
                        
                        return img, index + 1

    return img, 0


cap = init_webcam_camera(webcam_w, webcam_h)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)

    if start_menu_select_flag == 0:
        img, start_menu_select_flag = display_start_menu(img, start_menu_info, lmList, detector)
    
    ####################### Face Tracking Mode
    elif start_menu_select_flag == 1:
        
        me = TelloDrone_Connect()
        TelloDrone_takeOff(me)
        
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, bboxInfo = detector.findPosition(img)
        
            cv2.putText(img, "<Face Tracking Mode On>", (40, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            img, faceTracking_menu_select_flag = display_menu(img, faceTracking_menu, lmList, detector)
            
            tello_w, tello_h, pxError, pyError, fbRange, pid = tello_init()
            
            tello_image = me.get_frame_read().frame
            tello_image = cv2.resize(tello_image, (tello_w, tello_h))
            tello_image, info = findFace(tello_image)
            pxError, pyError = trackFace(info, tello_w, tello_h, pid, pxError, pyError, fbRange)
            
            if faceTracking_menu_select_flag == 1:
                faceTracking_menu_select_flag = 0
                start_menu_select_flag = 0
                me.land()
                break
            
            elif faceTracking_menu_select_flag == 2:
                date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M%S"))
                cv2.imwrite('WebcamPicture/' + str(date) + '.jpg', img)
                sleep(0.15)
                
            elif faceTracking_menu_select_flag == 3:
                date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M%S"))
                cv2.imwrite('TelloPicture/' + str(date) + '.jpg', tello_image)
                sleep(0.15)

            cv2.imshow("TelloDrone Program", img)
            cv2.imshow("FaceTracking", tello_image)
            cv2.waitKey(1)

    ####################### Face Tracking with Recognition Mode On
    elif start_menu_select_flag == 2:
        
        me = TelloDrone_Connect()
        TelloDrone_takeOff(me)
        
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, bboxInfo = detector.findPosition(img)
            
            cv2.putText(img, "<Face Tracking & Recognition Mode On>", (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            img, faceTrackingRecognition_menu_select_flag = display_menu(img, faceTrackingRecognition_menu, lmList, detector)
            
            tello_w, tello_h, pxError, pyError, fbRange, pid = tello_init()
            
            tello_image = me.get_frame_read().frame
            tello_image = cv2.resize(tello_image, (tello_w, tello_h))
            tello_image, info = findFaceWithRecognition(tello_image)
            pxError, pyError = trackFace(info, tello_w, tello_h, pid, pxError, pyError, fbRange)
            
            
            if faceTrackingRecognition_menu_select_flag == 1:
                faceTrackingRecognition_menu_select_flag = 0
                start_menu_select_flag = 0
                me.land()
                break
            
            elif faceTrackingRecognition_menu_select_flag == 2:
                date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M%S"))
                cv2.imwrite('WebcamPicture/' + str(date) + '.jpg', img)
                sleep(0.15)
                
            elif faceTrackingRecognition_menu_select_flag == 3:
                date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M%S"))
                cv2.imwrite('TelloPicture/' + str(date) + '.jpg', tello_image)
                sleep(0.15)
            
            cv2.imshow("TelloDrone Program", img)
            cv2.imshow("FaceTracking", tello_image)
            cv2.waitKey(1)

    ####################### Direct Control Mode On
    elif start_menu_select_flag == 3:
        
        me = TelloDrone_Connect()
        TelloDrone_takeOff(me)
        takeOff_flag = 1
        
        while True:
            
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, bboxInfo = detector.findPosition(img)
            
            cv2.putText(img, "<Direct Control Mode On>", (40, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            img, directControl_menu_select_flag = direct_control_display_menu(img, directControl_menu, lmList, detector)
            
            lr, fb, ud, yv = 0, 0, 0, 0
            speed = 30
            
            tello_w, tello_h, pxError, pyError, fbRange, pid = tello_init()
            
            tello_image = me.get_frame_read().frame
            tello_image = cv2.resize(tello_image, (tello_w, tello_h))
            
            if directControl_menu_select_flag == 1: #takeOff
                if takeOff_flag: 
                    me.land()
                    takeOff_flag = 0
                else: 
                    TelloDrone_takeOff(me)
                    takeOff_flag = 1
            
            elif directControl_menu_select_flag == 2: # up
                ud = speed
                
            elif directControl_menu_select_flag == 3: # DOWN
                ud = -speed
            
            elif directControl_menu_select_flag == 4: # LEFT
                lr = -speed
                
            elif directControl_menu_select_flag == 5: # RIGHT
                lr = speed
                
            elif directControl_menu_select_flag == 6: # FW
                fb = speed
                
            elif directControl_menu_select_flag == 7: # BW
                fb = -speed
                
            elif directControl_menu_select_flag == 9:
                date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M%S"))
                cv2.imwrite('WebcamPicture/' + str(date) + '.jpg', img)
                sleep(0.15)
                
            elif directControl_menu_select_flag == 10:
                date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M%S"))
                cv2.imwrite('TelloPicture/' + str(date) + '.jpg', tello_image)
                sleep(0.15)
            
            me.send_rc_control(lr, fb, ud, yv)
            
            if directControl_menu_select_flag == 8: # QUIT
                directControl_menu_select_flag = 0
                start_menu_select_flag = 0
                me.land()
                break    
            
            
            
            cv2.imshow("TelloDrone Program", img)
            cv2.imshow("Direct Mode", tello_image)
            cv2.waitKey(1)
    
    ####################### Making Dataset & Training & Recognition
    elif start_menu_select_flag == 4:
            
        while True:
            
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, bboxInfo = detector.findPosition(img)
            
            cv2.putText(img, "<Webcam Dataset Mode On>", (40, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

            
            img, webCamDataset_menu_select_flag = display_menu(img, webCamDataset_menu, lmList, detector)
            
            if webCamDataset_menu_select_flag == 1:
                making_dataset()     
            
            elif webCamDataset_menu_select_flag == 2:
                while True:
                    success, img = cap.read()
                    img = cv2.flip(img, 1)
                    img, _ = findFaceWithRecognition(img)
                    cv2.imshow("Recognition", img)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        webCamDataset_menu_select_flag = 0
                        cv2.destroyWindow("Recognition")
                        break
            
            elif webCamDataset_menu_select_flag == 3: # QUIT
                webCamDataset_menu_select_flag = 0
                start_menu_select_flag = 0
                break    
            
            cv2.imshow("TelloDrone Program", img)
            cv2.waitKey(1)
            
    
    elif start_menu_select_flag == 5:
        break    
    
    cv2.imshow("TelloDrone Program", img)
    cv2.waitKey(1)




    


# 추가하고 싶은 사항
# 1. recognition을 진행할 수 있는 웹 캠 데이터셋 모으기
# 1-2. 모은 데이터셋으로 얼굴 학습
# 1-3. recognition 수행

# 2. 촬영 기능

# 3. tello_image.release() 