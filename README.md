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
