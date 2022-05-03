from load_custom_model import loadCustomModel
import numpy as np
import cv2
import torch

if __name__ == "__main__":
    model = loadCustomModel(path='best.pt', conf=0.3, iou=0.5)
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # IMG DETECTION
    img = 'img (3).jpg'
    img = cv2.imread(img)[..., ::-1]
    results = model(img)
    results.print()
    results.show()
    cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])
    cv2.waitKey(0)

    # VIDEO DETECTION
    # cap = cv2.VideoCapture(0)
    # # cap = cv2.VideoCapture('input2.mp4')
    # while True:
    #     success, img = cap.read()
    #     img = img[..., ::-1]
    #     results = model(img)
    #     cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()