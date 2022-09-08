#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# This is an example of head pose estimation with solvePnP.
# It uses the dlib library and openCV
#

import numpy
import cv2
import sys

import sys, os
sys.path.append(os.path.abspath('./Tensorflow2.0-PFLD-'))
from glue import PFLD_TFLite, ULFace
from utils.landmark_utils import *
from utils.face_detection_utils import *
from copy import deepcopy
	
from scipy.spatial.transform import Rotation as R

#from deepgaze.haar_cascade import haarCascade
#from deepgaze.face_landmark_detection import faceLandmarkDetection


#If True enables the verbose mode
DEBUG = True 

#Antropometric constant values of the human head. 
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = np.array([0, 8, 16, 24, 32, 33, 46, 51, 54, 57, 60, 64, 68, 72, 90]).reshape(-1, 1)
ALL_POINTS = list(range(0,68)) #Used for debug only


def main():

	#Defining the video capture object
	video_capture = cv2.VideoCapture(0)

	if(video_capture.isOpened() == False):
		print("Error: the resource is busy or unvailable")
	else:
		print("The video source has been opened correctly...")

	#Create the main window and move it
	cv2.namedWindow('Video')
	cv2.moveWindow('Video', 20, 20)

	#Obtaining the CAM dimension
	cam_w = int(video_capture.get(3))
	cam_h = int(video_capture.get(4))

	#Defining the camera matrix.
	#To have better result it is necessary to find the focal
	# lenght of the camera. fx/fy are the focal lengths (in pixels) 
	# and cx/cy are the optical centres. These values can be obtained 
	# roughly by approximation, for example in a 640x480 camera:
	# cx = 640/2 = 320
	# cy = 480/2 = 240
	# fx = fy = cx/tan(60/2 * pi / 180) = 554.26
	c_x = cam_w / 2
	c_y = cam_h / 2
	f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
	f_y = f_x

	#Estimated camera matrix values.
	camera_matrix = numpy.float32([[f_x, 0.0, c_x],
								   [0.0, f_y, c_y], 
								   [0.0, 0.0, 1.0] ])

	print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

	#These are the camera matrix values estimated on my webcam with
	# the calibration code (see: src/calibration):
	#camera_matrix = numpy.float32([[602.10618226,		  0.0, 320.27333589],
	#							   [		 0.0, 603.55869786,  229.7537026], 
	#							   [		 0.0,		  0.0,		  1.0] ])

	#Distortion coefficients
	#camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

	#Distortion coefficients estimated by calibration
	#camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])
	camera_distortion = None

	#This matrix contains the 3D points of the
	# 11 landmarks we want to find. It has been
	# obtained from antrophometric measurement
	# on the human head.
	landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
								  P3D_GONION_RIGHT,
								  P3D_MENTON,
								  P3D_GONION_LEFT,
								  P3D_LEFT_SIDE,
								  P3D_FRONTAL_BREADTH_RIGHT,
								  P3D_FRONTAL_BREADTH_LEFT,
								  P3D_SELLION,
								  P3D_NOSE,
								  P3D_SUB_NOSE,
								  P3D_RIGHT_EYE,
								  P3D_RIGHT_TEAR,
								  P3D_LEFT_TEAR,
								  P3D_LEFT_EYE,
								  P3D_STOMION])

	#Declaring the two classifiers
	face_detector = ULFace.Detector()
	#TODO If missing, example file can be retrieved from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	landmark_predictor = PFLD_TFLite.Predictor()

	#Error counter definition
	no_face_counter = 0

	#Variables that identify the face
	#position in the main frame.
	face_x1 = 0
	face_y1 = 0
	face_x2 = 0
	face_y2 = 0
	face_w = 0
	face_h = 0

	#Variables that identify the ROI
	#position in the main frame.
	roi_x1 = 0
	roi_y1 = 0
	roi_x2 = cam_w
	roi_y2 = cam_h
	roi_w = cam_w
	roi_h = cam_h
	roi_resize_w = int(cam_w/10)
	roi_resize_h = int(cam_h/10)
	
	landmarks =  np.empty( shape=(0, 0) )

	while(True):

		# Capture frame-by-frame
		ret, frame = video_capture.read()
		if not ret: break
		
		height, width = frame.shape[:2]
		
		is_landmarks_detected = landmarks.size != 0
	
		is_face_detected, bbox = face_detector.detect_bbox(frame)
		
		if is_face_detected:
			bbox = face_detector.post_process(bbox)
	
			xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
		
			cv2.rectangle(frame, (xmin, ymin),
					(xmax, ymax), (125, 255, 0), 2)
				
			img = crop(frame, bbox)
	
			landmarks = landmark_predictor.predict(img)
			
			landmarks_2D = landmark_predictor.post_process(landmarks, bbox)
			for (x, y) in landmarks_2D:
				cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
			
			#TRACKED_POINTS = np.array(TRACKED_POINTS).reshape(-1, 1)
			landmarks_2D = landmarks_2D[TRACKED_POINTS]
			landmarks_2D = np.float32([item[0] for item in landmarks_2D])
			print(landmarks_2D)
			for (x, y) in landmarks_2D:
				cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (0, 0, 255))
			#Applying the PnP solver to find the 3D pose
			# of the head from the 2D position of the
			# landmarks.
			#retval - bool
			#rvec - Output rotation vector that, together with tvec, brings 
			# points from the model coordinate system to the camera coordinate system.
			#tvec - Output translation vector.
			retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
											  landmarks_2D, 
											  camera_matrix, camera_distortion)

			#Now we project the 3D points into the image plane
			#Creating a 3-axis to be used as reference in the image.
			axis = numpy.float32([[50,0,0], 
								  [0,50,0], 
								  [0,0,50]])
			imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

			#Drawing the three axis on the image frame.
			#The opencv colors are defined as BGR colors such as: 
			# (a, b, c) >> Blue = a, Green = b and Red = c
			#Our axis/color convention is X=R, Y=G, Z=B
			sellion_xy = (np.int32(landmarks_2D[7][0]), np.int32(landmarks_2D[7][1]))
			cv2.line(frame, sellion_xy, tuple(np.int32(imgpts[1].ravel())), (0,255,0), 3) #GREEN
			cv2.line(frame, sellion_xy, tuple(np.int32(imgpts[2].ravel())), (255,0,0), 3) #BLUE
			cv2.line(frame, sellion_xy, tuple(np.int32(imgpts[0].ravel())), (0,0,255), 3) #RED

		#Showing the frame and waiting
		# for the exit command
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
   
	#Release the camera
	video_capture.release()
	print("Bye...")


if __name__ == "__main__":
	main()
