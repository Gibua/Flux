import argparse
import cv2
import numpy as np
import copy
import tensorflow as tf

from scipy.special import softmax, expit
from functools import partial
from Model.datasets import DateSet
from scipy.special import softmax, expit
from Model.utils import parse_arguments, Normalization, color_


class Predictor:
	
	def __init__(self):	
		self.interpreter = tf.lite.Interpreter(model_path='../models/tflite/pfld_infer.tflite')
		self.interpreter.allocate_tensors()
		self.input_details = interpreter.get_input_details()
		self.output_details = interpreter.get_output_details()
		

	def set_input_tensor(interpreter, image):
		tensor_index = interpreter.get_input_details()[0]['index']
		print(interpreter.get_input_details())
		interpreter.set_tensor(tensor_index, image)
	
	def pre_process(img):
		imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(imgrgb, dsize=(112,112))
		image_rgb = resized[..., ::-1]
		image_norm = image_rgb.astype(np.float32)
		cv2.normalize(image_norm, image_norm,
			alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
		return image_norm[None, ...]
	
	def predict(img):
		input_image = pre_process(img).astype(np.float32)
		
		self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
		self.interpreter.invoke()
		
		landmarks = self.interpreter.get_tensor(self.output_details[0]['index'])
		landmarks = expit(landmarks)
		landmarks = landmarks.reshape(-1, 2)
		
		return landmarks
		
	def post_process(landmarks, bbox):
		bbox_width = bbox[0]-bbox[2]
		for point in landmarks:
			point[0] = (point[0]*bbox_width)+bbox[0]
			point[1] = (point[1]*bbox_width)+bbox[1]

