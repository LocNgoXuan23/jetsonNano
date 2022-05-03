import cv2
import numpy as np
import face_recognition
import os
from my_utils import getConfig, updateConfig
import time
from handle_encode_data import updateEncodeList

# currentMember = getConfig()['currentMember']
# encodeListKnown, classNames = updateEncodeList()
# print('Encoding Complete !!')

def myFaceRecognition(img, encodeListKnown, classNames):
	imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
	imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

	facesCurFrame = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

	for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
		matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
		faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
		# print(faceDis)
		matchIndex = np.argmin(faceDis)

		if matches[matchIndex]:
			name = classNames[matchIndex].upper()
			print(name)
		else:
			name = 'Undefine'
			print(name)

		y1, x2, y2, x1 = faceLoc
		y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
		cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
	return img

# cap = cv2.VideoCapture(0)
# state = 0
# numOfSample = 10
# currentSample = 0

# while True:
# 	success, img = cap.read()
# 	if state == 0:
# 		img = myFaceRecognition(img)
# 	if state == 1:
# 		newPath = os.path.join('data', f'member_{currentMember - 1}', f'{currentSample}.member_{currentMember - 1}.png')
# 		cv2.imwrite(newPath, img)
# 		print(f'Read Sample {currentSample}')
# 		time.sleep(1)
# 		currentSample += 1
# 		if currentSample == numOfSample:
# 			currentSample = 0
# 			state = 0
# 			encodeListKnown, classNames = updateEncodeList()

# 	if cv2.waitKey(1) & 0xFF == ord('1'):
# 		state = 1
# 		newPath = os.path.join('data', f'member_{currentMember}')
# 		if os.path.exists(newPath):
# 			pass
# 		else:
# 			os.mkdir(newPath)
# 			updateConfig()
# 			currentMember = getConfig()['currentMember']
# 		time.sleep(3)


# 	if cv2.waitKey(1) & 0xFF == ord('0'):
# 		state = 0



# 	cv2.imshow('Webcam', img)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

