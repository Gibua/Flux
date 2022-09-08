import argparse
import cv2
import numpy as np
import sys, os
import copy
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

from rt_gene.estimate_gaze_tensorflow import GazeEstimator


class Estimator():
	
	
	def __init__(self):
		self._script_path = os.path.dirname(os.path.realpath(__file__))
		relative_path = './modules/TFLitePFLD/models/tflite/pfld_infer.tflite'
		models = [os.path.abspath(os.path.join(self._script_path, '../rt_gene/model_nets/Model_allsubjects1.h5'))]
		
		self.gaze_estimator = GazeEstimator("/gpu:0", models)
		
	@property
	def numberoflandmarks(self):
		return self._numberoflandmarks
	
	def pre_process(self, img):
		pass
		
	def extract_eye_image_patches(img, landmarks, bbox, eye_corner_indices):
		re_c, le_c, _, _ = get_eye_image_from_landmarks(img, landmarks, bbox, eye_corner_indices)
		return re_c, le_c
		
	def rvec_to_theta_phi(self, rvec):
		_rotation_matrix, _ = cv2.Rodrigues(rvec)
		_rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
		_m = np.zeros((4, 4))
		_m[:3, :3] = _rotation_matrix
		_m[3, 3] = 1
		# Go from camera space to ROS space
		_camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
						  [-1.0, 0.0, 0.0, 0.0],
						  [0.0, -1.0, 0.0, 0.0],
						  [0.0, 0.0, 0.0, 1.0]]
		roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
		roll_pitch_yaw = limit_yaw(roll_pitch_yaw)
		
		phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)
		
		return phi, theta
	
	def predict(self, img, bbox, landmarks, eye_corner_indices, rvec):
		right_eye, left_eye = extract_eye_image_patches(img, bbox, landmarks, eye_corner_indices)
		
		if left_eye is None or right_eye is None:
			return
		
		phi_head, theta_head = rvec_to_theta_phi(rvec)

		face_image_resized = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
		
		input_r_list.append(gaze_estimator.input_from_image(right_eye))
		input_l_list.append(gaze_estimator.input_from_image(left_eye))
		input_head_list.append([theta_head, phi_head])
		valid_subject_list.append(0)
		
		gaze_est = self.gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
														inference_input_right_list=input_r_list,
														inference_headpose_list=input_head_list)
		
		return gaze_est.tolist()[0], right_eye, left_eye
		
	def build_visualization(self, gaze, right_eye, left_eye):
		r_gaze_img = self.gaze_estimator.visualize_eye_result(right_eye, gaze)
		l_gaze_img = self.gaze_estimator.visualize_eye_result(left_eye, gaze)
		s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)
		return cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB)
		
	def post_process(self, landmarks, bbox):
		pass
			
	
	def get_eye_image_from_landmarks(self, img, bbox, landmarks, eye_corner_indices, eye_image_size=(60, 36)):
		eye_landmarks = landmarks[eye_corner_indices]
		eye_landmarks[:, 0] -= bbox[0]
		eye_landmarks[:, 1] -= bbox[1]
		
		margin_ratio = 1.0
		desired_ratio = float(eye_image_size[1]) / float(eye_image_size[0]) / 2.0

		try:
			# Get the width of the eye, and compute how big the margin should be according to the width
			lefteye_width = eye_landmarks[3][0] - eye_landmarks[2][0]
			righteye_width = eye_landmarks[1][0] - eye_landmarks[0][0]

			lefteye_center_x = eye_landmarks[2][0] + lefteye_width / 2
			righteye_center_x = eye_landmarks[0][0] + righteye_width / 2
			lefteye_center_y = (eye_landmarks[2][1] + eye_landmarks[3][1]) / 2.0
			righteye_center_y = (eye_landmarks[1][1] + eye_landmarks[0][1]) / 2.0

			aligned_face, rot_matrix = GenericTracker.align_face_to_eyes(img, 
										right_eye_center=(righteye_center_x, righteye_center_y),
										left_eye_center=(lefteye_center_x, lefteye_center_y))

			# rotate the eye landmarks by same affine rotation to extract the correct landmarks
			ones = np.ones(shape=(len(eye_landmarks), 1))
			points_ones = np.hstack([eye_landmarks, ones])
			transformed_eye_landmarks = rot_matrix.dot(points_ones.T).T

			# recompute widths, margins and centers
			lefteye_width = transformed_eye_landmarks[3][0] - transformed_eye_landmarks[2][0]
			righteye_width = transformed_eye_landmarks[1][0] - transformed_eye_landmarks[0][0]
			lefteye_margin, righteye_margin = lefteye_width * margin_ratio, righteye_width * margin_ratio
			lefteye_center_y = (transformed_eye_landmarks[2][1] + transformed_eye_landmarks[3][1]) / 2.0
			righteye_center_y = (transformed_eye_landmarks[1][1] + transformed_eye_landmarks[0][1]) / 2.0

			# Now compute the bounding boxes
			# The left / right x-coordinates are computed as the landmark position plus/minus the margin
			# The bottom / top y-coordinates are computed according to the desired ratio, as the width of the image is known
			left_bb = np.zeros(4, dtype=np.int)
			left_bb[0] = transformed_eye_landmarks[2][0] - lefteye_margin / 2.0
			left_bb[1] = lefteye_center_y - (lefteye_width + lefteye_margin) * desired_ratio
			left_bb[2] = transformed_eye_landmarks[3][0] + lefteye_margin / 2.0
			left_bb[3] = lefteye_center_y + (lefteye_width + lefteye_margin) * desired_ratio

			right_bb = np.zeros(4, dtype=np.int)
			right_bb[0] = transformed_eye_landmarks[0][0] - righteye_margin / 2.0
			right_bb[1] = righteye_center_y - (righteye_width + righteye_margin) * desired_ratio
			right_bb[2] = transformed_eye_landmarks[1][0] + righteye_margin / 2.0
			right_bb[3] = righteye_center_y + (righteye_width + righteye_margin) * desired_ratio

			# Extract the eye images from the aligned image
			left_eye_color = aligned_face[left_bb[1]:left_bb[3], left_bb[0]:left_bb[2], :]
			right_eye_color = aligned_face[right_bb[1]:right_bb[3], right_bb[0]:right_bb[2], :]

			# So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
			left_eye_color_resized = cv2.resize(left_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)
			right_eye_color_resized = cv2.resize(right_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)

			return right_eye_color_resized, left_eye_color_resized, right_bb, left_bb
		except (ValueError, TypeError, cv2.error) as e:
			return None, None, None, None

