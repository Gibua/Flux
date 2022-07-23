import argparse
import cv2
import time
import numpy as np
import os
import sys
import copy

sys.path.append(os.path.abspath('./Tensorflow2.0-PFLD-'))
sys.path.append(os.path.abspath('./SADRNet'))

import config

from glue import PFLD_TFLite, ULFace, SADRNet
from utils.landmark_utils import *
from utils.face_detection_utils import *

# from glue.PFLD_TFLite import *

#sys.path.insert(1, '/glue')
#import glue

landmark_predictor = SADRNet.Predictor()
face_detector = ULFace.Detector()

cap = cv2.VideoCapture(0)

landmarks = np.empty( shape=(0, 0) )
bbox = None
bbox_prev = None
last_detection = None
is_face_detected = False

i = 0

while True:
	ret, frame = cap.read()
	if not ret: break
	
	height, width = frame.shape[:2]
	
	frame_crop = None
	
	is_landmarks_detected = landmarks.size != 0
	
	if (i == 0) or (i%20 == 0):
		is_face_detected, last_detection = face_detector.detect_bbox(frame)
		if is_face_detected and (not is_landmarks_detected):
			bbox = last_detection.copy()
			bbox_prev = last_detection
	if (i != 0) and is_face_detected and is_landmarks_detected:
		landmark_bbox = bbox_from_landmark(landmarks)
		
		last_detection_area = bbox_area(last_detection)
		intersect_area = bbox_area(bbox_intersect(last_detection, landmark_bbox))
		intersect_proportion = intersect_area/last_detection_area
			
		# print(intersect_proportion)
			
		if (intersect_proportion<0.5):
			is_face_detected, last_detection = face_detector.detect_bbox(frame)
			if is_face_detected:
				bbox = last_detection.copy()
				bbox_prev = last_detection
		else:
			bbox = bboxes_average(landmark_bbox, bbox_prev)
			bbox_prev = last_detection
	
	if is_face_detected:
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
