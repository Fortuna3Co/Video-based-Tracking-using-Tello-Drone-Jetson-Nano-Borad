import cv2
import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()

img_w, img_h = 360, 240

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (img_w, img_h))