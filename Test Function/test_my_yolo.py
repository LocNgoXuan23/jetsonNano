# import the opencv library
import cv2
import torch
# import tensorflow
from load_custom_model import loadCustomModel
import numpy as np
import time

model = loadCustomModel(path='best.pt', conf=0.4, iou=0.7)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

dispW=1280
dispH=720
flip=4
pTime = 0
camSet='nvarguscamerasrc wbmode=1 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=20/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=2 brightness=-.1 saturation=1.2 ! appsink'
# camSet = 'input1.mp4'
cap = cv2.VideoCapture(camSet)
while(True):
	success, img = cap.read()

	cTime = time.time()
	fps = 1 / (cTime - pTime)
	pTime = cTime
	cv2.putText(img, f'{int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
				(255, 0, 0), 5)

	img = img[..., ::-1]
	results = model(img)
	cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
  
cap.release()
cv2.destroyAllWindows()
