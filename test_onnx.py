import argparse
import cv2
import time
import numpy as np
import os
import sys
import copy

import onnxruntime

sys.path.append(os.path.abspath('./modules/Tensorflow2.0-PFLD-'))
sys.path.append(os.path.abspath('./modules/SADRNet'))

import config

from glue import PFLD_TFLite, ULFace, SADRNet
from utils.landmark import *
from utils.face_detection import *

# from glue.PFLD_TFLite import *

#sys.path.insert(1, '/glue')
#import glue
from src.dataset.dataloader import img_to_tensor
def predict(model, img):
	resized = cv2.resize(img, dsize=(256,256))
	image = (resized / 255.0).astype(np.float32)
	for ii in range(3):
		image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
			image[:, :, ii].var() + 0.001)
	tensor = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
	input_image = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
	

landmark_predictor = onnxruntime.InferenceSession("./SADRNv2.onnx", None)
print(landmark_predictor.get_inputs()[0].name)
print(landmark_predictor.get_outputs()[0].name)

face_detector = ULFace.Detector()

cap = cv2.VideoCapture(0)

landmarks = np.empty( shape=(0, 0) )
bbox = None
bbox_prev = None
last_detection = None
is_face_detected = False

i = 0

while True:
	ret, frame = cap.read()
	if not ret: break
			
	cv2.imshow('1', frame)
	
	i = i+1
	
	if cv2.waitKey(1) == 27:
		break
	
cap.release()
cv2.destroyAllWindows()
