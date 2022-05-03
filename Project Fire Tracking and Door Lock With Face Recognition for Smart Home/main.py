import sys
sys.path.insert(1, 'Fire-Tracking')
sys.path.insert(2, 'Face-Recognition')
from load_custom_model import loadCustomModel
from face_recognition_module import myFaceRecognition
import cv2
import numpy as np
import face_recognition
import os
from my_utils import getConfig, updateConfig
import time
from handle_encode_data import updateEncodeList
import numpy as np
import cv2

FIRE_TRACKING_PATH = 'Fire-Tracking'
FACE_RECOGNITION_PATH = 'Face-Recognition'

model = loadCustomModel(path=f'{FIRE_TRACKING_PATH}/best.pt', conf=0.4, iou=0.7)

# # IMG DETECTION
# img = f'{FIRE_TRACKING_PATH}/img (2).jpg'
# img = cv2.imread(img)[..., ::-1]
# results = model(img)
# results.print()
# results.show()
# cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])
# cv2.waitKey(0)

# # VIDEO DETECTION
# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture(f'{FIRE_TRACKING_PATH}/input1.mp4')
# while True:
#     success, img = cap.read()
#     img = img[..., ::-1]
#     results = model(img)
#     cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

currentMember = getConfig(os.path.join(FACE_RECOGNITION_PATH, 'data', 'config.yml'))['currentMember']
encodeListKnown, classNames = updateEncodeList(os.path.join(FACE_RECOGNITION_PATH, 'data'))
print('Encoding Complete !!')

cap = cv2.VideoCapture(0)
state = -1
numOfSample = 10
currentSample = 0

while True:
	success, img = cap.read()
	# For tracking Fire
	if state == -1:
		print('tracking fire')
		img = img[..., ::-1]
		results = model(img)
		img = np.squeeze(results.render())[..., ::-1]

	# For Face Recognition
	if state == 0:
		img = myFaceRecognition(img, encodeListKnown, classNames)
	# For Add new Member
	if state == 1:
		newPath = os.path.join(FACE_RECOGNITION_PATH, 'data', f'member_{currentMember - 1}', f'{currentSample}.member_{currentMember - 1}.png')
		cv2.imwrite(newPath, img)
		print(f'Read Sample {currentSample}')
		time.sleep(1)
		currentSample += 1
		if currentSample == numOfSample:
			currentSample = 0
			state = 0
			encodeListKnown, classNames = updateEncodeList(os.path.join(FACE_RECOGNITION_PATH, 'data'))

	if cv2.waitKey(1) & 0xFF == ord('1'):
		state = 1
		newPath = os.path.join(FACE_RECOGNITION_PATH, 'data', f'member_{currentMember}')
		if os.path.exists(newPath):
			pass
		else:
			os.mkdir(newPath)
			updateConfig(os.path.join(FACE_RECOGNITION_PATH, 'data', 'config.yml'))
			currentMember = getConfig(os.path.join(FACE_RECOGNITION_PATH, 'data', 'config.yml'))['currentMember']
		time.sleep(3)

	if cv2.waitKey(1) & 0xFF == ord('0'):
		state = 0

	cv2.imshow('Webcam', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()