import numpy as np
import cv2
import torch

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

    # IMG DETECTION
    img = '1.jpg'
    img = cv2.imread(img)[..., ::-1]
    results = model(img)
    results.print()
    results.show()
    cv2.imshow('YOLO', np.squeeze(results.render())[..., ::-1])
    # cv2.imshow('YOLO', img)
    cv2.waitKey(0)