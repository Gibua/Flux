import argparse
import cv2
import time
from math import sqrt

def crop(image, bbox):
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
	
def average(number1, number2):
	return (number1 + number2) / 2.0
