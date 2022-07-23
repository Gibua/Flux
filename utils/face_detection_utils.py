import argparse
import cv2
import time
import numpy as np
from math import sqrt

def crop(frame, bbox):
	image = frame.copy()
	xmin = max(bbox[0], 0)
	ymin = max(bbox[1], 0)
	xmax = min(bbox[2], image.shape[1])
	ymax = min(bbox[3], image.shape[0])
	cut = image[ymin:ymax, xmin:xmax,:]
	
	return cut
	
def square_box(result):
	ymin = result[1]
	ymax = result[3]
	
	xmin = result[0]
	xmax = result[2]
	
	height = ymax-ymin
	width = xmax-xmin
	
	
	if (height>width):
		center = xmin+((xmax-xmin)/2)
		xmin = int(center-height/2)
		xmax = int(center+height/2)
	else:
		center = ymin+((ymax-ymin)/2)
		ymin = int(center-width/2)
		ymax = int(center+width/2)	
	
	return [xmin, ymin, xmax, ymax]
	
	
def bbox_from_landmark(landmarks):
	xmin = min(landmarks[:, 0])
	ymin = min(landmarks[:, 1])
	xmax = max(landmarks[:, 0])
	ymax = max(landmarks[:, 1])
	bbox = [xmin, ymin, xmax, ymax]
	center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
	radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
	bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

	llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
	center_x = (bbox[2] + bbox[0]) / 2
	center_y = (bbox[3] + bbox[1]) / 2

	bbox = [0] * 4
	bbox[0] = center_x - llength / 2
	bbox[1] = center_y - llength / 2
	bbox[2] = bbox[0] + llength
	bbox[3] = bbox[1] + llength
	
	return bbox
	
def bbox_area(bbox):
	if len(bbox) == 0:
		return 0
	else:
		return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def bbox_intersect(bbox_current, bbox_prev):
	if (bbox_current[0] > bbox_prev[2] or bbox_prev[0] > bbox_current[2]) \
		or (bbox_current[1] > bbox_prev[3] or bbox_prev[1] > bbox_current[3]):
		return None
	intersection = np.empty(4)
	intersection[0] = max(bbox_current[0], bbox_prev[0])
	intersection[1] = max(bbox_current[1], bbox_prev[1])
	intersection[2] = min(bbox_current[2], bbox_prev[2])
	intersection[3] = min(bbox_current[3], bbox_prev[3])
	return intersection
	
def int_average(number1, number2):
	return int(round((number1 + number2) / 2.0))
	
def bboxes_average(bbox1, bbox2):
	bbox = [int_average(bbox1[0], bbox2[0]),
			int_average(bbox1[1], bbox2[1]),
			int_average(bbox1[2], bbox2[2]),
			int_average(bbox1[3], bbox2[3])]
		
	return bbox

def unwrap_bbox(bbox):
	xmin = bbox[0]
	ymin = bbox[1]
	xmax = bbox[2]
	ymax = bbox[3]
		
	return xmin, ymin, xmax, ymax
	
def average(number1, number2):
	return (number1 + number2) / 2.0
