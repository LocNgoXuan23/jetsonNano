import cv2
#from load_custom_model import loadCustomModel
import numpy as np
import time

#model = loadCustomModel(path='best.pt', conf=0.4, iou=0.7)
dispW=640
dispH=480
flip=4
pTime=0
# camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
camSet='nvarguscamerasrc wbmode=1 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=20/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=2 brightness=-.1 saturation=1.2 ! appsink'

vid = cv2.VideoCapture(camSet)

  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #frame = frame[..., ::-1]
    #results = model(frame)
    #cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])
    
    cTime = time.time()
    print(cTime)
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'{int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.imshow('frame', frame)
    #time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
