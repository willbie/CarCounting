import cv2
import numpy as np
from tracker import *

tracker = EuclideanDistTracker()
cap = cv2.VideoCapture("hour1.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

kernel = np.ones((4,4),np.uint8)

while True:
  ret, frame = cap.read()
  # height, width, _ = frame.shape
  # print(height, width)
  #object detection

  # roi = frame[0:200, 300:600]

  mask = object_detector.apply(frame)

  _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

  dilated = cv2.dilate(mask,kernel,iterations = 1)

  contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


  detections = []
  for cnt in contours:
    #caculate area
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if (area > 2000) & (x >= 300) & (x <= 350) & (y <= 100) :
      # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
      # x, y, w, h = cv2.boundingRect(cnt)
      # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
      detections.append([x, y, w, h])

  boxes_ids = tracker.update(detections)
  for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

  # cv2.imshow("Roi", roi)
  cv2.imshow("Frame", frame)
  # cv2.imshow("Mask", mask)
  # cv2.imshow("Dilated", dilated)

  key = cv2.waitKey(30)
  if (key == 27):
    break

cap.release()
cv2.destroyAllWindows