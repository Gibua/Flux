"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Head Pose Euler Angles(pitch yaw roll) estimation from 2D-3D correspondences landmarks
"""

import cv2
import numpy as np

class EulerAngles:
	"""
		Head Pose Estimation from landmarks annotations with solvePnP OpenCV
		Pitch, Yaw, Roll Rotation angles (euler angles) from 2D-3D Correspondences Landmarks

		Givin a General 3D Face Model (3D Landmarks) & annotations 2D landmarks
		2D point = internsic * exterinsic * 3D point_in_world_space
		if we have 2D - 3D correspondences & internsic camera matrix, 
		we can use cv2.solvPnP to get the extrinsic matrix that convert the world_space to camera_3D_space
		this extrinsic matrix is considered as the 3 euler angles & translation vector

		we can do that because the faces have similar 3D structure and emotions & iluminations dosn't affect the pose
		Notes:
			we can choose get any 3D coord from any 3D face model .. changing translation will not affect the angle
			it will only inroduce a bigger tvec but the same rotation matrix
	"""

	def __init__(self,  img_shape=(112,112) ):
		# Lazy Estimation of Camera internsic Matrix Approximation
		self.camera_intrinsic_matrix = self.estimate_camera_matrix(img_shape)		
		# 3D Face model 3D landmarks
		self.landmarks_3D = self.get_face_model_3D_landmarks()

	def estimate_camera_matrix(self, img_shape):
		# Used Weak Prespective projection as we assume near object with similar depths

		# cx, cy the optical centres
		# translation to image center as image center here is top left corner
		# focal length is function of image size & Field of View (assumed to be 30 degree)
		c_x = img_shape[0] / 2
		c_y = img_shape[1] / 2
		FieldOfView = 60
		focal = c_x / np.tan(np.radians(FieldOfView/2))
		
		# Approximated Camera intrinsic matrix assuming weak prespective
		return np.float32([
			[focal, 0.0,	c_x], 
			[0.0,   focal,  c_y],
			[0.0,   0.0,	1.0]
		])

	def set_img_shape(self, img_shape):
		self.camera_intrinsic_matrix = self.estimate_camera_matrix(img_shape)

	def get_face_model_3D_landmarks(self):
		"""
			General 3D Face Model Coordinates (3D Landmarks) 
			obtained from antrophometric measurement of the human head.

			Returns:
			-------
			3D_Landmarks: numpy array of shape(N, 3) as N = 11 point in 3D
		"""
		# X-Y-Z with X pointing forward and Y on the left and Z up (same as LIDAR)
		# OpenCV Coord X points to the right, Y down, Z to the front (same as 3D Camera)
		
		landmarks_3D = np.float32([
			[0.05832857, -7.60688782, -15.69785643],  # LEFT_EYEBROW_LEFT, 
			[-0.10377414, 8.64764404, -22.13105822],  # LEFT_EYEBROW_RIGHT, 
			[-0.24472262, 23.37460136, -31.45995176],  # RIGHT_EYEBROW_LEFT,
			[-0.37655398, 35.09374356, -36.40296197],  # RIGHT_EYEBROW_RIGHT,
			[-12.36220503, 46.71208954, -14.98328161],  # LEFT_EYE_LEFT,
			[7.14245749, 49.0878067, -18.29475546],  # LEFT_EYE_RIGHT,
			[-0.20381828, 50.27907181, -19.5088048],  # RIGHT_EYE_LEFT,
			[6.80970144, 49.12047958, -18.35101175],  # RIGHT_EYE_RIGHT,
			[12.06582785, 46.73353767, -15.02448034],  # NOSE_LEFT,
			[-46.25580263, -1.64848328, 2.87227488],  # NOSE_RIGHT,
			[-19.0150342, 1.5302906, -3.10160017],  # MOUTH_LEFT,
			[19.16842031, 1.49678993, -2.83671331],  # MOUTH_RIGHT,
			[46.10241652, -1.37859726, 3.06603861]  # LOWER_LIP,
		])
		
		for i in range(len(landmarks_3D)):
			for j in range(3):
				landmarks_3D[i][j] *= 0.15
		print(landmarks_3D)
		return landmarks_3D


	def euler_angles_from_landmarks(self, landmarks_2D):
		"""
			Estimates Euler angles from 2D landmarks 
			
			Parameters:
			----------
			landmarks_2D: numpy array of shape(N, 2) as N is num of landmarks (usualy 98 from WFLW)

			Returns:
			-------
			rvec: rotation numpy array that transform model space to camera space (3D in both)
			tvec: translation numpy array that transform model space to camera space
			euler_angles: (pitch yaw roll) in degrees
		"""

		# WFLW(98 landmark) tracked points
		TRACKED_POINTS_MASK = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 68, 72]
		landmarks_2D = landmarks_2D[TRACKED_POINTS_MASK]

		"""
			solve for extrinsic matrix (rotation & translation) with 2D-3D correspondences
			returns:
				rvec: rotation vector (as rotation is 3 degree of freedom, it is represented as 3d-vector)
				tvec: translate vector (world origin position relative to the camera 3d coord system)
				_ : error -not important-.
		"""
		_, rvec, tvec = cv2.solvePnP(self.landmarks_3D, landmarks_2D, self.camera_intrinsic_matrix, distCoeffs=None)

		"""
			note:
				tvec is almost constant = the world origin coord with respect to the camera
				avarage value of tvec = [-1,-2,-21]
				we can use this directly without computing tvec
		"""

		# convert rotation vector to rotation matrix .. note: function is used for vice versa
		# rotation matrix that transform from model coord(object model 3D space) to the camera 3D coord space
		rotation_matrix, _ = cv2.Rodrigues(rvec)

		# [R T] may be used in cv2.decomposeProjectionMatrix(extrinsic)[6]
		extrinsic_matrix = np.hstack((rotation_matrix, tvec))	
		
		# decompose the extrinsic matrix to many things including the 3 euler angles 
		# (pitch yaw roll) in degrees
		euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]

		return rvec, tvec, euler_angles

def draw_euler_axis(image, axis_pts, euler_angles):
	"""
		draw euler axes in the image center 
	"""
	center = (image.shape[1]//2, image.shape[0]//2)

	axis_pts = axis_pts.astype(np.int32)
	pitch_point = tuple(axis_pts[0].ravel())
	yaw_point   = tuple(axis_pts[1].ravel())
	roll_point  = tuple(axis_pts[2].ravel())

	pitch_color = (255,255,0)
	yaw_color   = (0,255,0)
	roll_color  = (0,0,255)

	pitch, yaw, roll = euler_angles

	cv2.line(image, center,  pitch_point, pitch_color, 2)
	cv2.line(image, center,  yaw_point, yaw_color, 2)
	cv2.line(image, center,  roll_point, roll_color, 2)
	cv2.putText(image, "Pitch:{:.2f}".format(pitch), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
	cv2.putText(image, "Yaw:{:.2f}".format(yaw), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
	cv2.putText(image, "Roll:{:.2f}".format(roll), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

	# origin
	cv2.circle(image, center, 2, (255,255,255), -1)
	return image

def draw_euler_angles(image, rvec, tvec, euler_angles, intrinsic_matrix):
		# i, j, k axes in world 3D coord.
		axis = np.identity(3) * 5
		# axis_img_pts = intrinsic * exstrinsic * axis
		axis_pts = cv2.projectPoints(axis, rvec, tvec, intrinsic_matrix, None)[0]
		image = draw_euler_axis(image, axis_pts, euler_angles)

		return image
		
def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 255), line_width=2):
	"""Draw a 3D box as annotation of pose"""
	point_3d = []
	rear_size = 10
	rear_depth = 0
	point_3d.append((-rear_size, -rear_size, rear_depth))
	point_3d.append((-rear_size, rear_size, rear_depth))
	point_3d.append((rear_size, rear_size, rear_depth))
	point_3d.append((rear_size, -rear_size, rear_depth))
	point_3d.append((-rear_size, -rear_size, rear_depth))

	front_size = 12
	front_depth = 20
	point_3d.append((-front_size, -front_size, front_depth))
	point_3d.append((-front_size, front_size, front_depth))
	point_3d.append((front_size, front_size, front_depth))
	point_3d.append((front_size, -front_size, front_depth))
	point_3d.append((-front_size, -front_size, front_depth))
	point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

	# Map to 2d image points
	(point_2d, _) = cv2.projectPoints(point_3d,
									  rotation_vector,
									  translation_vector,
									  camera_matrix,
									  np.zeros((4, 1)))
	point_2d = np.int32(point_2d.reshape(-1, 2))

	# Draw all the lines
	cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
	cv2.line(image, tuple(point_2d[1]), tuple(
		point_2d[6]), color, line_width, cv2.LINE_AA)
	cv2.line(image, tuple(point_2d[2]), tuple(
		point_2d[7]), color, line_width, cv2.LINE_AA)
	cv2.line(image, tuple(point_2d[3]), tuple(
		point_2d[8]), color, line_width, cv2.LINE_AA)

if __name__ == "__main__":
	import sys, os
	sys.path.append(os.path.abspath('./Tensorflow2.0-PFLD-'))
	from glue import PFLD_TFLite, ULFace
	from utils.landmark_utils import *
	from utils.face_detection_utils import *
	from copy import deepcopy

	landmark_predictor = PFLD_TFLite.Predictor()
	face_detector = ULFace.Detector()
	euler_estimator = EulerAngles()

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print('camera not detected')
	
	landmarks = np.empty( shape=(0, 0) )
	bbox = None
	bbox_prev = None
	last_detection = None
	is_face_detected = False

	i = 0
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		
		height, width = frame.shape[:2]
		
		euler_estimator = EulerAngles((width, height))
		
		is_landmarks_detected = landmarks.size != 0
	
		is_face_detected, bbox = face_detector.detect_bbox(frame)
		
		if is_face_detected:
			bbox = face_detector.post_process(bbox)
	
			xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
		
			cv2.rectangle(frame, (xmin, ymin),
					(xmax, ymax), (125, 255, 0), 2)
				
			img = crop(frame, bbox)
	
			landmarks = landmark_predictor.predict(img)
			
			landmarks_for_drawing = landmark_predictor.post_process(landmarks, bbox)
			for (x, y) in landmarks_for_drawing:
				cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
				
			for point in [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 68, 72]:
				cv2.circle(frame, 
							(np.int32(landmarks_for_drawing[point][0]), np.int32(landmarks_for_drawing[point][1])),
							 1, (0, 0, 255))
				
			rvec, tvec, euler_angles = euler_estimator.euler_angles_from_landmarks(landmarks)
				
			#print(rvec)
			
			if i==0:
				rvec_cal = deepcopy(rvec)
				
			rvec -= rvec_cal
			
			euler_angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
			
			tvec[0] = 0
			tvec[1] = 0
			tvec[2] = -80
			
			#bbox_width = bbox[2]-bbox[0]
			#tvec = np.array([(bbox_width)/112, bbox_width/112, tvec[2]]).astype(np.float32)
			#tvec = euler_estimator.camera_intrinsic_matrix @ tvec_cal
			#print(tvec)
				
			
			#c_x = width / 2
			#c_y = height / 2
			#FieldOfView = 60
			#focal = c_x / np.tan(np.radians(FieldOfView/2))
			
			#tvec = np.array([-1,-2,-21]).reshape((3, 1)).astype(np.float32)
			#print(tvec.shape)
			
			#tvec[0] += bbox[0]
			#tvec[1] += bbox[1]
			
			#matrix = np.float32([[focal, 0.0,   c_x], 
			#					 [0.0,   focal, c_y],
			#					 [0.0,   0.0,   1.0]])
				
			frame = draw_euler_angles(frame, rvec, tvec, euler_angles, euler_estimator.camera_intrinsic_matrix)
			draw_annotation_box(frame, rvec, tvec, euler_estimator.camera_intrinsic_matrix)

		cv2.imshow('1', frame)
	
		i = i+1
	
		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
