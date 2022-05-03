import cv2
import numpy as np
import face_recognition

# dispW=640
# dispH=480
# flip=4
# camSet='nvarguscamerasrc wbmode=1 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=2 brightness=-.1 saturation=1.2 ! appsink'
# # camSet = 'input1.mp4'
# cap = cv2.VideoCapture(camSet)
  
# while(True):
#     success, img = cap.read()

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     faceLoc = face_recognition.face_locations(img)
#     cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
    
#     cv2.imshow('face recognition', img)

#     # cv2.imshow('YOLO', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
  
# cap.release()
# cv2.destroyAllWindows()

img = face_recognition.load_image_file('2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(img)[0]
cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
cv2.imshow('test', img)
cv2.waitKey(0)


