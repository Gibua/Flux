import argparse
import cv2
import time
from math import sqrt

def crop(image, xmin, ymin, xmax, ymax):
	xmin = max(xmin, 0)
	ymin = max(ymin, 0)
	xmax = min(xmax, image.shape[1])
	ymax = min(ymax, image.shape[0])
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
	
	
def parse_roi_box_from_landmark(pts):
	"""calc roi box from landmark"""
	xmin = min(pts[:, 0])
	ymin = min(pts[:, 1])
	xmax = max(pts[:, 0])
	ymax = max(pts[:, 1])
	bbox = [xmin, ymin, xmax, ymax]
	center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
	radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
	bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

	llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
	center_x = (bbox[2] + bbox[0]) / 2
	center_y = (bbox[3] + bbox[1]) / 2

	roi_box = [0] * 4
	roi_box[0] = center_x - llength / 2
	roi_box[1] = center_y - llength / 2
	roi_box[2] = roi_box[0] + llength
	roi_box[3] = roi_box[1] + llength
	
	print(ymax)
	print(roi_box)

	return roi_box
	
def average(number1, number2):
	return (number1 + number2) / 2.0
