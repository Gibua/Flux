import argparse
import cv2
import numpy as np
import sys, os
import copy
import tensorflow as tf

from scipy.special import softmax, expit
from functools import partial
from scipy.special import softmax, expit

from Model.utils import parse_arguments, Normalization, color_
from Model.datasets import DateSet

from .Detector import Detector


class Predictor(Detector):
	
	def __init__(self):
		relative_path = './Tensorflow2.0-PFLD-/models/tflite/pfld_infer.tflite'
		self.interpreter = tf.lite.Interpreter(model_path=os.path.abspath(relative_path))
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		

	def set_input_tensor(self, interpreter, image):
		tensor_index = interpreter.get_input_details()[0]['index']
		print(interpreter.get_input_details())
		interpreter.set_tensor(tensor_index, image)
	
	def pre_process(self, img):
		imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(imgrgb, dsize=(112,112))
		image_rgb = resized[..., ::-1]
		image_norm = image_rgb.astype(np.float32)
		cv2.normalize(image_norm, image_norm,
			alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
		return image_norm[None, ...]
	
	def predict(self, img):
		input_image = self.pre_process(img)
		
		self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
		self.interpreter.invoke()
		
		landmarks = self.interpreter.get_tensor(self.output_details[0]['index'])
		landmarks = expit(landmarks)
		landmarks = landmarks.reshape(-1, 2)
		
		return landmarks
		
	def post_process(self, landmarks, bbox):
		bbox_width = bbox[2]-bbox[0]
		for point in landmarks:
			point[0] = (point[0]*bbox_width)+bbox[0]
			point[1] = (point[1]*bbox_width)+bbox[1]
		return landmarks.copy()

