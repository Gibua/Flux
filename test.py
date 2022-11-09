import argparse
import cv2
import time
import numpy as np
import os
import sys
import copy

#sys.path.append(os.path.abspath('./modules/Tensorflow2.0-PFLD-'))
#sys.path.append(os.path.abspath('./modules/SADRNet'))

#import config

from glue import ULFace, PIPNet
from utils.landmark import *
from utils.face_detection import *

# from glue.PFLD_TFLite import *

#sys.path.insert(1, '/glue')
#import glue

landmark_predictor = PIPNet.Predictor("300W_COFW_WFLW")
face_detector = ULFace.Detector()

cap = cv2.VideoCapture(0)

landmarks = np.empty( shape=(0, 0) )
bbox = None
bbox_prev = None
last_detection = None
is_face_detected = False

i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    height, width = frame.shape[:2]

    is_landmarks_detected = landmarks.size != 0

    if (i == 0) or (i%50 == 0):
        is_face_detected, last_detection = face_detector.detect_bbox(frame)
        #print(is_face_detected)
        if is_face_detected and (not is_landmarks_detected):
            bbox = last_detection.copy()
            bbox_prev = last_detection
    if (i != 0) and is_face_detected and is_landmarks_detected:
        landmark_bbox = bbox_from_landmark(landmarks)
        print(landmark_bbox)

        intersection = bbox_intersect(last_detection, landmark_bbox)

        landmark_bbox_area = bbox_area(np.array(landmark_bbox))
        intersect_area = bbox_area(intersection)
        intersect_proportion = intersect_area/landmark_bbox_area

        print("----------",landmark_bbox_area)

        if (intersect_proportion<0.65):
            is_face_detected, last_detection = face_detector.detect_bbox(frame)
            #print(last_detection)
            if is_face_detected:
                bbox = last_detection.copy()
                bbox_prev = last_detection
        else:
            bbox_prev = bbox
            bbox = bboxes_average(landmark_bbox, bbox_prev)
    print(bbox)
    if is_face_detected:
        #bbox = face_detector.post_process(bbox)
        bbox = np.int32(bbox)
        xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
        
        cv2.rectangle(frame, (xmin, ymin),
                    (xmax, ymax), (125, 255, 0), 2)

        img = crop(frame, bbox)
    
        #start_time = time.perf_counter()
        preds = landmark_predictor.predict(frame, bbox)
        landmarks = landmark_predictor.post_process(preds, bbox)
        #print(time.perf_counter() - start_time)
        #print( np.array([np.array([112*(point[0]), 112*(point[1])]) for point in landmarks]) )
    
        for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (0, 0, 255))
            
    cv2.imshow('1', frame)
    
    i = i+1
    
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
