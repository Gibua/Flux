"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import cv2
import eos
import numpy as np

class EosHeadPoseEstimator(object):

	def __init__(self):
		cwd = os.path.dirname(__file__)
		base_dir = cwd + '/ext/eos'

		model = eos.morphablemodel.load_model(base_dir + '/share/sfm_shape_3448.bin')
		self.blendshapes = eos.morphablemodel.load_blendshapes(
			base_dir + '/share/expression_blendshapes_3448.bin')
		self.morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(
			model.get_shape_model(), self.blendshapes,
			eos.morphablemodel.PcaModel(),
			model.get_texture_coordinates(),
		)
		self.landmark_mapper = eos.core.LandmarkMapper(
			base_dir + '/share/wflw_to_sfm.txt')
		self.edge_topology = eos.morphablemodel.load_edge_topology(
			base_dir + '/share/sfm_3448_edge_topology.json')
		self.contour_landmarks = eos.fitting.ContourLandmarks.load(
			base_dir + '/share/wflw_to_sfm.txt')
		self.model_contour = eos.fitting.ModelContour.load(
			base_dir + '/share/sfm_model_contours.json')

	def fit_func(self, landmarks, image_size):
		image_w, image_h = image_size
		return eos.fitting.fit_shape_and_pose(
			self.morphablemodel_with_expressions, landmarks_to_eos(landmarks),
			self.landmark_mapper, image_w, image_h, self.edge_topology,
			self.contour_landmarks, self.model_contour,
		)


def landmarks_to_eos(landmarks):
	out = []
	for i, (x, y) in enumerate(landmarks[:68, :]):
		out.append(eos.core.Landmark(str(i + 1), [x, y]))
	return out


class PnPHeadPoseEstimator(object):
	wflw_ids_to_use = sorted([
		51, 52, 53, 54,  # nose ridge
		55, 56, 57, 58, 59,  # nose base
		60, 64,  # left-eye corners
		68, 72,  # right-eye corners
	])

	def __init__(self):
		# Load and extract vertex positions for selected landmarks
		cwd = os.path.dirname(__file__)
		base_dir = cwd + '/ext/eos'
		self.model = eos.morphablemodel.load_model(
			base_dir + '/share/sfm_shape_3448.bin')
		self.shape_model = self.model.get_shape_model()
		self.landmarks_mapper = eos.core.LandmarkMapper(
			base_dir + '/share/wflw_to_sfm.txt')
		self.sfm_points_wflw_subset = np.array([
			self.shape_model.get_mean_at_point(
				int(self.landmarks_mapper.convert(str(d)))
			)
			for d in range(1, 69)
			if self.landmarks_mapper.convert(str(d)) is not None
		])

		self.sfm_points_for_pnp = np.array([
			self.shape_model.get_mean_at_point(
				int(self.landmarks_mapper.convert(str(d)))
			)
			for d in self.wflw_ids_to_use
		])

		# Rotate face around
		rotate_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
		self.sfm_points_wflw_subset = np.matmul(self.sfm_points_wflw_subset.reshape(-1, 3), rotate_mat)
		self.sfm_points_for_pnp = np.matmul(self.sfm_points_for_pnp.reshape(-1, 3), rotate_mat)

		# Center on mean point between eye corners
		between_eye_point = np.mean(self.sfm_points_for_pnp[-4:, :], axis=0)
		self.sfm_points_wflw_subset -= between_eye_point.reshape(1, 3)
		self.sfm_points_for_pnp -= between_eye_point.reshape(1, 3)
		for i in range(len(self.sfm_points_for_pnp)):
			for j in range(3):
				self.sfm_points_for_pnp[i][j] *= 0.15
		print(self.sfm_points_for_pnp)

	def get_sfm_points_for_pnp(self):
		return self.sfm_points_for_pnp

	def fit_func(self, landmarks, camera_matrix):
		landmarks = np.array([
			landmarks[i - 1, :]
			for i in self.wflw_ids_to_use
		], dtype=np.float64)
		
		success, rvec, tvec, inliers = cv2.solvePnPRansac(self.sfm_points_for_pnp, landmarks,
														  camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)

		# Second fit for higher accuracy
		success, rvec, tvec = cv2.solvePnP(self.sfm_points_for_pnp, landmarks, camera_matrix, None,
										   rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
		
		euler_angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
		
		return rvec, tvec, euler_angles

	def project_model(self, rvec, tvec, camera_parameters):
		fx, fy, cx, cy = camera_parameters
		camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
		points, _ = cv2.projectPoints(self.sfm_points_wflw_subset, rvec, tvec, camera_matrix, None)
		return points


	def drawPose(self, img, r, t, cam, dist):

		modelAxes = np.array([
			np.array([0., -20., 0.]).reshape(1, 3),
			np.array([50., -20., 0.]).reshape(1, 3),
			np.array([0., -70., 0.]).reshape(1, 3),
			np.array([0., -20., -50.]).reshape(1, 3)
		])

		projAxes, jac = cv2.projectPoints(modelAxes, r, t, cam, dist)

		cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
				 (int(projAxes[1, 0, 0]), int(projAxes[1, 0, 1])),
				 (0, 255, 255), 2)
		cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
				 (int(projAxes[2, 0, 0]), int(projAxes[2, 0, 1])),
				 (255, 0, 255), 2)
		cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
				 (int(projAxes[3, 0, 0]), int(projAxes[3, 0, 1])),
				 (255, 255, 0), 2)
				  
def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 255), line_width=2):
	"""Draw a 3D box as annotation of pose"""
	point_3d = []
	rear_size = 100
	rear_depth = 0
	point_3d.append((-rear_size, -rear_size, rear_depth))
	point_3d.append((-rear_size, rear_size, rear_depth))
	point_3d.append((rear_size, rear_size, rear_depth))
	point_3d.append((rear_size, -rear_size, rear_depth))
	point_3d.append((-rear_size, -rear_size, rear_depth))

	front_size = 120
	front_depth = 200
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
	euler_estimator = PnPHeadPoseEstimator()

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
		
		c_x = width / 2
		c_y = height / 2
		FieldOfView = 60
		focal = c_x / np.tan(np.radians(FieldOfView/2))
			
		matrix = np.float64([[focal, 0.0,   c_x], 
							 [0.0,   focal, c_y],
							 [0.0,   0.0,   1.0]])
		
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
			#for (x, y) in landmarks_for_drawing:
			#	cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
				
			#for point in [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 68, 72]:
			#	cv2.circle(frame, 
			#				(np.int32(landmarks_for_drawing[point][0]), np.int32(landmarks_for_drawing[point][1])),
			#				 1, (0, 0, 255))
			
			#print(euler_estimator.get_sfm_points_for_pnp())

			rvec, tvec, euler_angles = euler_estimator.fit_func(landmarks_for_drawing, matrix)
				
			#print(rvec)
			
			if i==0:
				rvec_cal = deepcopy(rvec)
				
			rvec -= rvec_cal
			
			euler_angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
			
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
				
			#frame = draw_euler_angles(frame, rvec, tvec, euler_angles, euler_estimator.camera_intrinsic_matrix)
			draw_annotation_box(frame, rvec, tvec, matrix)
			euler_estimator.drawPose(frame, rvec, tvec, matrix, 10)
			
			pitch, yaw, roll = euler_angles
			pitch_color = (255,255,0)
			yaw_color   = (0,255,0)
			roll_color  = (0,0,255)
			cv2.putText(frame, "Pitch:{:.2f}".format(pitch), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
			cv2.putText(frame, "Yaw:{:.2f}".format(yaw), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
			cv2.putText(frame, "Roll:{:.2f}".format(roll), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

		cv2.imshow('1', frame)
	
		i = i+1
	
		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
