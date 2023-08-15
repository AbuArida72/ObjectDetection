import numpy as np
from ultralytics import YOLO
import time
import cv2
import cvzone
import math
from sort import *

width = 1080
height = 720
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.1.180:8080/video")
cap = cv2.VideoCapture("car file/phone.mp4")
cap.set(3, 640)
cap.set(4, 640)

model = YOLO("../Yolo-Wights/yolov8l.pt")
totalCount = []
totalCountDown = []
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "tie",
              "handbag", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
              "hair drier", "toothbrush"
              ]
mask = cv2.imread("mask/r.png")
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# line points two points
limits = [520, 740, 1270, 740]

#writer = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

while True:
    success, img = cap.read()

    imgGraphic = cv2.imread("car file/counter2.png", cv2.IMREAD_UNCHANGED)
    imgGraphic = cv2.resize(imgGraphic, (200, 100))
    img = cv2.resize(img, (width, height))
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img = cvzone.overlayPNG(img, imgGraphic, (0, 0))

    # imgRegion = cv2.bitwise_and(img, mask_resized)
    results = model(img, stream=True)

    detections = np.empty((0, 5))
    #writer.write(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box----------------------------------------------------------
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)

            # conference-------------------------------------------------------------
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if (currentClass == "person" or currentClass == "cell phone") and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=8,rt=5)
                cvzone.putTextRect(img, f'{classNames[cls]}{conf}',
                                   (max(0, x1), max(40, y1)),
                                   scale=0.8,
                                   thickness=1,
                                   offset=3
                                   )
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
            # print(x1, y1, x2, y2)
    resultsTracker = tracker.update(detections)
    # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=5, colorR=(255, 0, 0))
        # cvzone.putTextRect(img, f'{int(id)}',
        #                    (max(0, x1), max(40, y1)),
        #                    scale=2,
        #                    thickness=3,
        #                    offset=10
        #                    )
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    ##########################################################################################################################
    # if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
    #     print("the line")
    #     if totalCount.count(id) == 0:
    #         totalCount.append(id)
    #         cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 5)

    cv2.putText(img, str(len(resultsTracker)), (120, 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
    # cvzone.putTextRect(img, f'{len(totalCount)}',
    #                    (50, 50),
    #                    scale=2,
    #                    thickness=3,
    #                    offset=10
    #                    )

    cv2.imshow("cam1", img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        while not cv2.waitKey(1) & 0xFF == ord('q'):
            continue


cap.release()
#writer.release()
cv2.destroyAllWindows()
