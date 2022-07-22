import argparse
import cv2
import time
import numpy as np
import os
import sys
import copy

sys.path.append(os.path.abspath('./Tensorflow2.0-PFLD-'))

from glue import PFLD_TFLite, ULFace
from utils.landmark_utils import *
from utils.face_detection_utils import *



# from glue.PFLD_TFLite import *

#sys.path.insert(1, '/glue')
#import glue

landmark_predictor = PFLD_TFLite.Predictor()
face_detector = ULFace.Detector()

cap = cv2.VideoCapture(0)

landmarks = None
bbox = None
bbox_prev = None
is_face_detected = False

i = 0

while True:
	#print("\n\n-------------------------------------------------\n\n")
	ret, frame = cap.read()
	#print(frame)
	if not ret: break
	
	height, width = frame.shape[:2]
	
	frame_crop = None
	result = None
	
	if (i==0) or (i%30==0):
		is_face_detected, bbox = face_detector.detect_bbox(frame)
		
		if(i != 0):
			bbox = bboxes_average(bbox, bbox_prev)
	else:
		if is_face_detected:
			bbox = bbox_from_landmark(landmarks)
			bbox_prev = bbox.copy()
	
	
	bbox = face_detector.post_process(bbox)
	
	xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
		
	cv2.rectangle(frame, (xmin, ymin),
				(xmax, ymax), (125, 255, 0), 2)
				
	img = crop(frame, bbox)
	
	start_time = time.perf_counter()
	landmarks = landmark_predictor.predict(img)
	print(time.perf_counter() - start_time)
	#print(landmarks)
	
	landmarks = landmark_predictor.post_process(landmarks, bbox)
	
	for (x, y) in landmarks:
			cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (0, 0, 255))
			
	cv2.imshow('1', frame)
	
	i = i+1
	
	if cv2.waitKey(1) == 27:
		break
	
cap.release()
cv2.destroyAllWindows()
