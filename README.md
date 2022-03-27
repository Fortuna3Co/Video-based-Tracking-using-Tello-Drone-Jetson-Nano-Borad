# Video-based-Tracking-using-Tello-Drone-Jetson-Nano-Borad

### 파일 설명
- Dataset face Training.py : 지정한 폴더의 사진을 학습해 id를 부여한다. 학습한 모델은 trainer.yml로 저장된다. (id는 사진의 이름에 따라 지정 됨)

- DroneProgram.py : Face Tracking, Recognition, Virtual Controller를 종합한 최종 프로그램  

- Jetbot_Fullbody_Tracking.ipynb : Yahboom의 Jetbot AI Robot car를 조종하기 위해 OpenCV를 이용 (Robot Car 참고 : http://www.yahboom.net/study/JETBOT)

- TelloDrone_FaceTracking.py : Tello Drone을 이용해 FaceTracking 수행. 드론 와이파이와 젯슨 나노를 연결해야 정상 작동. (Tello Drone 참고 : https://www.ryzerobotics.com/)

- TelloDrone_FaceTracking_With_Recognition.py : 드론을 통해 받아온 영상정보를 Dataset face Training.py 를 통해 저장된 trainer.yml을 이용해 얼굴 인식 수행 (얼굴 인식을 제외한 기능은 TelloDrone_FaceTracking.py와 동일)
 
- VirtualKeyboard.py : 웹캠을 통한 가상 키보드 생성, 검지와 중지 사이 거리가 일정 거리 이하가 되면 클릭이라 판단

- WebCamDataset.py : 웹캠을 통해 학습시킬 얼굴을 저장하는 프로그램. 

- custom face recognition : 노트북 웹캠 화면에서 얼굴 인식을 판별할 수 있는 프로그램

- haarcascade_frontalface_default.xml, haarcascade_fullbody.xml : OpenCV에서 미리 학습된 haarcascade 파일을 사용했다.

(* 웹캠의 경우 노트북 내장 또는 Jetson Nano Board와 웹캠 카메라를 연결하여 사용한다.)

### 목표
- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)을 이용해 Tello Drone, Jetbot AI Robot Car의 영상 정보를 처리하고 OpenCV를 이용하여 Tracking 기술을 구현한다.


### Tello Drone - Jetson Nano Board 진행 과정

1. Jetson Nano Board의 경우 jetson nano developer kit b01 4gb를 사용했다.
2. Ubuntu 18.04 LTS를 이용하였으며 wifi 모듈을 장착하여 wifi를 사용했다.
3. CUDA 환경을 설정하여 GPU를 사용할 수 있게 설정했다. [(OpenCV cuda 설정)](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html)
4. [JETPACK](https://developer.nvidia.com/embedded/jetpack) 설정. 설정 완료 시 아래와 같은 화면을 볼 수 있다. (OpenCV : 4.5.4 complied CUDA : YES)
![OpenCV CUDA Yes](https://user-images.githubusercontent.com/78258412/160235026-d1588727-2dd0-475a-8410-9ad10c9b4dfb.png)
5. Jetson Nano Board에 [vscode 및 python 설치](https://www.jetsonhacks.com/2019/10/01/jetson-nano-visual-studio-code-python/)
6. 얼굴 추적 프로그램 작성 TelloDrone_FaceTracking - 드론이 얼굴을 인식할 경우 일정 거리를 유지한 채 따라감
``` python
import cv2
import numpy as np
from djitellopy import tello
import time
```

``` python
me = tello.Tello()      # tello 객체 생성
me.connect()            # tello 연결
print(me.get_battery()) # 연결된 드론의 현재 배터리 출력
```
연결 시 Jetson Nano Board와 Tello Drone은 같은 네트워크에 있어야 한다.(내부 네트워크 환경 구성 필요)

``` python
me.streamon()           # 영상 정보 출력
me.takeoff()            # 이륙
```

``` python
w, h = 800, 600             # 드론 영상 크기 조절
fbRange = [10000, 11000]    # 드론과 사람의 거리 조절
pid = [0.2, 0.2, 0]         #
px_Error, py_Error = 0, 0

while True:
    img = me.get_frame_read().frame     # 드론이 촬영하고 있는 영상 정보 수신
    img = cv2.resize(img, (w, h))       # OpenCV를 이용하여 영상 크기 조절 (영상 크기가 클 수록 성능 저하)
    img, info = findFace(img)           # 이미지에서 얼굴을 인식하고 위치를 반환함
    px_Error, py_Error = trackFace(info, w, pid, px_Error, py_Error) # findFace를 이용해 얻은 위치를 기반으로 드론 조종
    cv2.imshow("Output", img)           # 드론 화면 송출
    if cv2.waitKey(1) & 0xFF == ord('q'):       # q를 누를경우 착륙 및 프로그램 종료
        print("land")
        me.land()
        break

print("\n [INFO] Exiting Program and cleanup stuff")
img.release()
cv2.destroyAllWindows()
```

``` python
def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # 얼굴을 인식하기 위해 OpenCV의 haarcascade_frontalface_default.xml 파일을 이용
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8, minSize = (int(360 * 0.05), int(240 * 0.05)))     # 얼굴 인식 위치 반환
    myFaceListC = []             # 얼굴 인식 중앙 위치 반환
    myFaceListArea = []          # 얼굴 인식 크기
    # 다수의 얼굴이 인식될 경우 크기가 가장 큰 얼굴을 기반으로 데이터를 반환 

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    
    if len(myFaceListArea) != 0:    # 인식된 얼굴이 있을 경우
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:                           # 인식된 얼굴이 없을 경우
        return img, [[0, 0], 0]
``` python
def trackFace(info, w, pid, px_Error, py_Error):    # 얼굴 중앙 위치를 기반으로 Tracking
    area = info[1]
    x, y = info[0]
    fb = 0

    x_error = x - w // 2
    y_error = y - h // 2

    x_speed = pid[0] * x_error + pid[1] * (x_error - px_Error)
    x_speed = int(np.clip(x_speed, -100, 100))

    y_speed = pid[0] * y_error + pid[1] * (y_error - py_Error)
    y_speed = int(np.clip(y_speed, -100, 100))

    print("area : ", area)  # 얼굴이 인식된 크기 (가까울 수록 화면에 차지하는 비율이 많아져 area 값은 높아지고, 멀어질 수록 화면에 차지하는 비율이 낮아져 area 값은 낮아진다.)
    
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20    # 드론 속도
    elif area < fbRange[0] and area != 0:
        fb += 20    # 드론 속도
    
    
    if x == 0:
        x_speed = 0
        x_error = 0
        y_speed = 0
        y_error = 0
    
    
    me.send_rc_control(0, fb, -y_speed, x_speed)
    # fb : 앞, 뒤 속도
    # y_speed : 상, 하 속도
    # x_speed : yaw 속도
    return x_error, y_error # 이전과 현재의 위치 차이를 속도에 반영
```
7. 얼굴 분별 모델을 생성하기 위해 학습 데이터 생성 DroneCamDataste - 드론 카메라를 이용하여 얼굴이 인식될 경우 해당 area만 추출하여 이미지를 저장한다. (노트북 카메라를 이용할 경우 WebCamDataset을 사용)

``` python
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#cap = cv2.VideoCapture(0) # 노트북 웹 캠 데이터 이용 또는 보드와 연결된 카메라 이용

w, h = 360, 240
cap.set(3, w)
cap.set(4, h)

face_id = input('Input face ID : ') # face id를 입력받음
# id는 이후 리스트에서 인덱스로 사용되기 때문에 1부터 순차적으로 생성하는 것을 권장

me = tello.Tello()
me.connect()

me.streamon()       # 드론 카메라 데이터 이용


count = 0
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

    for (x, y, w, h) in faces:      # 얼굴이 인식될 경우 현재 경로 폴더에 dataset폴더 생성 이후 얼굴 데이터 저장
        cv2.rectangle (img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        date = str(datetime.datetime.now(timezone("UTC")).astimezone(timezone("Asia/Seoul")).strftime("%m%d%H%M"))
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + '.' + date + '.jpg', gray[y:y+h,x:x+w])
        print("dataset/User." + str(face_id) + '.' + str(count) + '.' + date + '.jpg', " was stored!")
        roi_gray = gray[y : y+h, x : x+w]
        roi_color = img[y : y+h, x : x+w]
    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xFF  # esc를 누르거나 120장의 데이터가 모아지면 종료
    if k == 27:
        break
    elif count >= 120:
        break

cap.release()
cv2.destroyAllWindows()
```

8. 얼굴 분별 모델 생성 Dataset face Training - 위에서 생성한 데이터들을 이용하여 LBPHFaceRecognizer로 학습시킨다. 학습된 모델은 trainer폴더에 trainer.yml 파일로 저장한다.

``` python
path = 'dataset'    # 학습 데이터가 존재하는 폴더 상대 경로
recognizer = cv2.face.LBPHFaceRecognizer_create()
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
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids)))
```

9. 얼굴 분별 적용 TelloDrone_FaceTracking_With_Recognition - Face tracking에 얼굴 분별만 할 수 있는 기능을 추가

``` python
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # 생성한 모델 불러오기

id = 0
names = ['None', 'KHS', 'KSY', 'LDS']   # 모델 생성시 id별로 이니셜 추가
font = cv2.FONT_HERSHEY_SIMPLEX         # 화면에 띄울 폰트 지정

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
me.takeoff()

w, h = 800, 600
fbRange = [10000, 11000]
pid = [0.1, 0.1, 0]
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
        
        if (confidence < 100):      # 인식 시 확률 및 아이디 출력
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"          # 미인식 시 unknown 출력
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
    
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]
```

10. OpenCV를 이용해서 가상 키보드 구현 VirtualKeyboard - 카메라를 이용한 가상 키보드 구현 예제

11. OpenCV를 이용해서 가상 컨트롤러 구현 TelloDrone_Directcontrol - 카메라를 이용해 Drone을 직접 조종할 수 있는 컨트롤러를 구현한다.

``` python
class display_menu():
    def __init__(self, pos, text, size=[800, 40]):
        self.pos = pos
        self.text = text
        self.size = size
        
def display_menu(img, menu_info, lmList, detector):
    
    for index, menu in enumerate(menu_info, 0):
        x, y = menu.pos
        w, h = menu.size
    
        cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        
        if lmList:
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    
                    l, _, _, = detector.findDistance(8, 12, img, draw=False)
                    # l의 경우 검지와 중지 사이의 거리를 의미한다.

                    if l < 50:  # 검지와 중지 사이의 거리가 일정이상 가까워 질 경우 클릭 수행
                        cv2.putText(img, menu.text, menu.pos, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        print('Clicked : ', index + 1)
                        
                        return img, index + 1

    return img, 0

directControl_menu = []

directControl_menu.append(display_menu[40, 600], 'Quit this mode')
directControl_menu.append(display_menu[50, 200], 'TakeOff')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)
    
    cv2.putText(img, "<Direct Control Mode On>", (40, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
    
    cv2.putText(img, "TakeOff", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.circle(img, (110, 190), 80, (0, 0, 0))
    
    cv2.imshow("Direct Control", img)
    cv2.waitKey(1)
```

![1130123517](https://user-images.githubusercontent.com/78258412/160264094-f2a3e032-6028-4aa8-a312-1dc2eed590f7.jpg)


12. Face tracking, Face Recognition, Direct Contorl을 종합해 하나의 프로그램을 구성 DroneProgram - 위에서 작성한 프로그램들을 하나의 프로그램으로 재구성, 카메라 촬영 기능 등 부가적인 기능을 추가했다.

### 보완 사항
* 일부 기능 들에 대해서 마무리가 되어 있지 않은 상태이며 현재 Drone과 Jetson Nano Board를 학교에 반납한 상태여서 추가적인 개선을 할 수 없는 상황이다.

* 연결을 유지하면서 이륙 / 착륙을 할 경우 정상적으로 작동하는 경우가 있는 반면 연결이 끊기는 경우도 있었다. 네트워크 문제라 추측된다.

* 얼굴 인식을 수행할 경우 빛의 세기에 따라 얼굴 인식율이 달라지는 것을 확인했다. 인식을 더 정밀하게 할 수 있도록 개선해야 한다.

* Face tracking, Direct control 등을 통해 드론을 조종할 경우 수신받는 카메라 화면이 불안정한 현상이 생겼다. Jetson Nano Board의 성능과 네트워크 연결의 불안정성 문제라 추측된다.



