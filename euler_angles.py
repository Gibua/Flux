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
								   #[-7.308957,  0.913869, 0.000000], #  0 JAWLINE_RIGHT
								   #[ 7.308957,  0.913869, 0.000000], #  1 JAWLINE_LEFT
								   [ 6.825897,  6.760612, 4.402142], #  2 LEFT_EYEBROW_LEFT,
								   [ 1.330353,  7.122144, 6.903745], #  3 LEFT_EYEBROW_RIGHT, 
								   [-1.330353,  7.122144, 6.903745], #  4 RIGHT_EYEBROW_LEFT,
								   [-6.825897,  6.760612, 4.402142], #  5 RIGHT_EYEBROW_RIGHT,
								   [ 5.311432,  5.485328, 3.987654], #  6 LEFT_EYE_LEFT,
								   [ 1.789930,  5.393625, 4.413414], #  7 LEFT_EYE_RIGHT,
								   [-1.789930,  5.393625, 4.413414], #  8 RIGHT_EYE_LEFT,
								   [-5.311432,  5.485328, 3.987654], #  9 RIGHT_EYE_RIGHT,
								   [ 1.930245,  0.424351, 5.914376], # 10 NOSE_LEFT,
								   [ 0.000000,  0.000000, 6.763430], # 11 NOSE_CENTER,
								   [-1.930245,  0.424351, 5.914376], # 12 NOSE_RIGHT,
								   [ 0.000000,  1.916389, 7.700000], # 13 NOSE_TIP,
								   #[ 2.774015, -2.080775, 5.048531], # 14 MOUTH_LEFT,
								   #[-2.774015, -2.080775, 5.048531], # 15 MOUTH_RIGHT,
								   #[ 0.000000, -1.646444, 6.704956], # 16 UPPER_LIP_CENTER,
								   #[ 0.000000, -7.415691, 4.070434], # CHIN
								  ])
		
		#landmarks_3D = np.array([
		#						(0.0, 0.0, 0.0),		  # Nose tip
		#						(0.0, -330.0, -65.0),	 # Chin
		#						(-225.0, 170.0, -135.0),  # Left eye left corner
		#						(225.0, 170.0, -135.0),	  # Right eye right corne
		#						(-150.0, -150.0, -125.0), # Left Mouth corner
		#						(150.0, -150.0, -125.0)	  # Right mouth corner
		#						])
		
		#Antropometric constant values of the human head.
		#Found on wikipedia and on:
		# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
		#
		#X-Y-Z with X pointing forward and Y on the left.
		#The X-Y-Z coordinates used are like the standard
		#landmarks_3D = np.array([
		#coordinates			   wflw point  description
		# (-100.0, -77.5,   -5.0),  # 0		 right side
		# (-110.0, -77.5,  -85.0),  # 8		 gonion right
		# (   0.0,   0.0, -122.7),  # 16		chin
		# (-110.0,  77.5,  -85.0),  # 24		gonion left
		# (-100.0,  77.5,   -5.0),  # 32		left side
		# ( -20.0, -56.1,   10.0),  # 33		frontal breadth right
		# ( -20.0,  56.1,   10.0),  # 46		frontal breadth left
		# (   0.0,   0.0,	0.0),  # 51		sellion
		# (  21.1,   0.0,  -48.0),  # 54		nose tip
		# (   5.0,   0.0,  -52.0),  # 57		sub nose
		# ( -20.0, -65.5,   -5.0),  # 60		right eye outer corner
		# ( -10.0, -40.5,   -5.0),  # 64		right eye inner corner
		# ( -10.0,  40.5,   -5.0),  # 68		left eye inner corner
		# ( -20.0,  65.5,   -5.0),  # 72		left eye outer corner
		# (  10.0,   0.0,  -75.0),  # 90		stomion
		#]).astype(np.float32)

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
		
		TRACKED_POINTS_MASK = [33, 38, 50, 46, 60, 64, 68, 72, 55, 57, 59, 54]
		#TRACKED_POINTS_MASK = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
		#TRACKED_POINTS_MASK = [0, 8, 16, 24, 32, 33, 46, 51, 54, 57, 60, 64, 68, 72, 90]
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

	front_size = 10
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
	
	from scipy.spatial.transform import Rotation as R

	landmark_predictor = PFLD_TFLite.Predictor()
	face_detector = ULFace.Detector()
	

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print('camera not detected')
	ret, frame = cap.read()
	euler_estimator = EulerAngles(frame.shape[:2])
	
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
				
			#for point in [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]:
			for point in [46, 50, 38, 33, 72, 68, 64, 60, 59, 57, 55, 54, 82, 76, 79]:
				cv2.circle(frame, 
							(np.int32(landmarks_for_drawing[point][0]), np.int32(landmarks_for_drawing[point][1])),
							 1, (0, 0, 255))
				
			rvec, tvec, euler_angles = euler_estimator.euler_angles_from_landmarks(landmarks_for_drawing)
				
			#print(rvec)
			
			if i==0:
				rvec_cal = deepcopy(rvec)
				
			rvec -= rvec_cal
			
			euler_angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
			#print(rvec)
			#print('\/')
			calc_from_euler = R.from_euler('xyz', [euler_angles[0],euler_angles[1],euler_angles[2]], 
												   degrees=True).as_matrix()
			#print(cv2.RQDecomp3x3(calc_from_euler)[0])
			#print(cv2.Rodrigues(calc_from_euler)[0])
			#print("-------------------------------------")
			
			#bbox_width = bbox[2]-bbox[0]
			#tvec = np.array([(bbox_width)/112, bbox_width/112, tvec[2]]).astype(np.float32)
			#tvec = euler_estimator.camera_intrinsic_matrix @ tvec_cal
			#print(tvec)
				
			
			c_x = width / 2
			c_y = height / 2
			FieldOfView = 60
			focal = c_x / np.tan(np.radians(FieldOfView/2))
			
			camera_matrix = np.float32([[focal, 0.0,   c_x], 
								 [0.0,   focal, c_y],
								 [0.0,   0.0,   1.0]])
			
			#tvec = np.array([-1,-2,-21]).reshape((3, 1)).astype(np.float32)
			#print(tvec.shape)
			
			#tvec[0] += bbox[0]
			#tvec[1] += bbox[1]
			
			landmarks_3D = np.float32([
			[-7.308957,  0.913869, 0.000000], #  0 JAWLINE_RIGHT
			[ 7.308957,  0.913869, 0.000000], #  1 JAWLINE_LEFT
			[ 6.825897,  6.760612, 4.402142], #  2 LEFT_EYEBROW_LEFT,
			[ 1.330353,  7.122144, 6.903745], #  3 LEFT_EYEBROW_RIGHT, 
			[-1.330353,  7.122144, 6.903745], #  4 RIGHT_EYEBROW_LEFT,
			[-6.825897,  6.760612, 4.402142], #  5 RIGHT_EYEBROW_RIGHT,
			[ 5.311432,  5.485328, 3.987654], #  6 LEFT_EYE_LEFT,
			[ 1.789930,  5.393625, 4.413414], #  7 LEFT_EYE_RIGHT,
			[-1.789930,  5.393625, 4.413414], #  8 RIGHT_EYE_LEFT,
			[-5.311432,  5.485328, 3.987654], #  9 RIGHT_EYE_RIGHT,
			[ 1.930245,  0.424351, 5.914376], # 10 NOSE_LEFT,
			[ 0.000000,  0.000000, 6.763430], # 11 NOSE_CENTER,
			[-1.930245,  0.424351, 5.914376], # 12 NOSE_RIGHT,
			[ 0.000000,  1.916389, 7.700000], # 13 NOSE_TIP,
			[ 2.774015, -2.080775, 5.048531], # 14 MOUTH_LEFT,
			[-2.774015, -2.080775, 5.048531], # 15 MOUTH_RIGHT,
			[ 0.000000, -1.646444, 6.704956], # 16 UPPER_LIP_CENTER,
			])
			
			landmarks_3D = np.float32([np.array([point[0], 
												 point[1]*(-1), 
												 point[2]]) for point in landmarks_3D])
			
			#TRACKED_POINTS = np.array([33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]).reshape(-1, 1)
			TRACKED_POINTS = np.array([32, 0, 33, 38, 50, 46, 60, 64, 68, 72, 55, 57, 59, 54, 76, 82, 79]).reshape(-1, 1)
			landmarks_2D = landmarks_for_drawing[TRACKED_POINTS]
			landmarks_2D = np.float32([item[0] for item in landmarks_2D])
			
			#print(landmarks_2D)
			
			#retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
			#								  landmarks_2D, 
			#								  camera_matrix, distCoeffs=None)
			
			
			#Now we project the 3D points into the image plane
			#Creating a 3-axis to be used as reference in the image.
			axis = np.float32([[50,0,0], 
							   [0,50,0], 
							   [0,0,50]])
			imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)
			
			object_pts_source = [[-7.308957,  0.913869,  0.000000], #  0  start jawline - right
								 [-6.775290, -0.730814, -0.012799],
								 [-5.665918, -3.286078,  1.022951],
								 [-5.011779, -4.876396,  1.047961],
								 [-4.056931, -5.947019,  1.636229],
								 [-1.833492, -7.056977,  4.061275],
								 [ 0.000000, -7.415691,  4.070434], #  6  chin center
								 [ 1.833492, -7.056977,  4.061275],
							 	 [ 4.056931, -5.947019,  1.636229],
							 	 [ 5.011779, -4.876396,  1.047961],
							 	 [ 5.665918, -3.286078,  1.022951],
							 	 [ 6.775290, -0.730814, -0.012799],
							 	 [ 7.308957,  0.913869,  0.000000], # 12  end jawline - left
							 	 [ 5.311432,  5.485328,  3.987654], # 13  start left eye - left corner
							 	 [ 4.461908,  6.189018,  5.594410],
							 	 [ 3.550622,  6.185143,  5.712299],
							 	 [ 2.542231,  5.862829,  4.687939],
							 	 [ 1.789930,  5.393625,  4.413414], # 17  left eye - right corner
							 	 [ 2.693583,  5.018237,  5.072837],
							 	 [ 3.530191,  4.981603,  4.937805],
							 	 [ 4.490323,  5.186498,  4.694397], # 20  end left eye
							 	 [-5.311432,  5.485328,  3.987654], # 21  right eye - right corner
							 	 [-4.461908,  6.189018,  5.594410],
							 	 [-3.550622,  6.185143,  5.712299],
							 	 [-2.542231,  5.862829,  4.687939],
							 	 [-1.789930,  5.393625,  4.413414], # 25  start right eye - left corner
							 	 [-2.693583,  5.018237,  5.072837], # 26  end right eye
							 	 [-3.530191,  4.981603,  4.937805],
								 [-4.490323,  5.186498,  4.694397],
								 [ 1.330353,  7.122144,  6.903745], # 29  end left eyebrow - right corner
								 [ 2.533424,  7.878085,  7.451034],
								 [ 4.861131,  7.878672,  6.601275],
							 	 [ 6.137002,  7.271266,  5.200823],
							 	 [ 6.825897,  6.760612,  4.402142], # 33  start left eyebrow - left corner
							 	 [-1.330353,  7.122144,  6.903745], # 34  start right eyebrow - left corner
								 [-2.533424,  7.878085,  7.451034],
								 [-4.861131,  7.878672,  6.601275],
								 [-6.137002,  7.271266,  5.200823],
								 [-6.825897,  6.760612,  4.402142], # 38  end right eyebrow - right corner
								 [-2.774015, -2.080775,  5.048531], # 39  mouth - right corner
							 	 [-0.509714, -1.571179,  6.566167],
							 	 [ 0.000000, -1.646444,  6.704956],
							 	 [ 0.509714, -1.571179,  6.566167],
							 	 [ 2.774015, -2.080775,  5.048531], # 43  start mouth - left corner
							 	 [ 0.589441, -2.958597,  6.109526], # 44  end mouth
							 	 [ 0.000000, -3.116408,  6.097667],
							 	 [-0.589441, -2.958597,  6.109526],
							 	 [-0.981972,  4.554081,  6.301271],
								 [-0.973987,  1.916389,  7.654050],
								 [-2.005628,  1.409845,  6.165652],
								 [-1.930245,  0.424351,  5.914376], # 50  end nose - right
								 [-0.746313,  0.348381,  6.263227],
							 	 [ 0.000000,  0.000000,  6.763430], # 52  nose bottom - center
								 [ 0.746313,  0.348381,  6.263227],
								 [ 1.930245,  0.424351,  5.914376], # 54  start nose - left
								 [ 2.005628,  1.409845,  6.165652],
								 [ 0.973987,  1.916389,  7.654050],
								 [ 0.981972,  4.554081,  6.301271],
								 [ 0.000000,  1.916389,  7.700000], # 58  nose tip
								]
			
			object_pts_source = np.float32([np.array([point[0], 
													  point[1]*(-1), 
													  point[2]]) for point in object_pts_source])
			mmins = [min(0, minimal) for minimal in np.amin(object_pts_source, axis=0)]
			
			print(mmins)
			
			object_pts_source = np.float32([np.float32([(point[0]-mmins[0])*20, 
														(point[1]-mmins[1])*20, 
														(point[2]-mmins[2])*20]) for point in object_pts_source])

			
			for (x, y, z) in object_pts_source:
				cv2.circle(frame, (np.int32(x), np.int32(y)), 3, (125, 255, 0))
				
			landmarks_3D  = np.float32([np.array([point[0]     , 
												  point[1], 
												  point[2]]     ) for point in landmarks_3D])
			#mmins = [min(0, minimal) for minimal in np.amin(landmarks_3D, axis=0)]
			
			print(mmins)
			
			landmarks_3D = np.float32([np.float32([(point[0]-mmins[0])*20, 
												   (point[1]-mmins[1])*20, 
												   (point[2]-mmins[2])*20]) for point in landmarks_3D])

			
			for (x, y, z) in landmarks_3D:
				cv2.circle(frame, (np.int32(x), np.int32(y)), 2, (0, 0, 255))
				
			for k in [41]:
				x, y, z = object_pts_source[k]
				cv2.circle(frame, (np.int32(x), np.int32(y)), 3, (0, 0, 255))

			#Drawing the three axis on the image frame.
			#The opencv colors are defined as BGR colors such as: 
			# (a, b, c) >> Blue = a, Green = b and Red = c
			#Our axis/color convention is X=R, Y=G, Z=B
			sellion_xy = (np.int32(landmarks_2D[7][0]), np.int32(landmarks_2D[7][1]))
			#cv2.line(frame, sellion_xy, tuple(np.int32(imgpts[1].ravel())), (0,255,0), 3) #GREEN
			#cv2.line(frame, sellion_xy, tuple(np.int32(imgpts[2].ravel())), (255,0,0), 3) #BLUE
			#cv2.line(frame, sellion_xy, tuple(np.int32(imgpts[0].ravel())), (0,0,255), 3) #RED
				
			#frame = draw_euler_angles(frame, rvec, tvec, euler_angles, euler_estimator.camera_intrinsic_matrix)
			draw_annotation_box(frame, rvec, tvec, camera_matrix, color=(0, 255, 0))

		cv2.imshow('1', frame)
	
		i = i+1
	
		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
