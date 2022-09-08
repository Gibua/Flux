import sys, os
import cv2
import numpy as np
import torch
import time
import torchvision.transforms as transforms

from model import Model

from modules.UltraLightFace.utils import *
from modules.UltraLightFace.TFLiteFaceDetector import UltraLightFaceDetecion

from utils.face_detection import *

from modules.OneEuroFilter import OneEuroFilter

def square_box(result):
	ymin = result[1]
	ymax = result[3]
	
	xmin = result[0]
	xmax = result[2]
	
	height = ymax-ymin
	width = xmax-xmin
	
	if (height>width):
		center = xmin+((xmax-xmin)/2)
		xmin = int(round(center-height/2))
		xmax = int(round(center+height/2))
	else:
		center = ymin+((ymax-ymin)/2)
		ymin = int(round(center-width/2))
		ymax = int(round(center+width/2))	
	
	return np.array([xmin, ymin, xmax, ymax])
	
def gazeto3d(gaze):
	assert gaze.size == 2, "The size of gaze must be 2"
	gaze_gt = np.zeros([3])
	gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
	gaze_gt[1] = -np.sin(gaze[1])
	gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
	return gaze_gt

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print('camera not detected')
	
	i = 0
	
	relative_path = "./modules/UltraLightFace/pretrained/version-RFB-320_without_postprocessing.tflite"
	detector = UltraLightFaceDetecion(os.path.abspath(relative_path), conf_threshold=0.5)
	GazeTR = Model()
	
	modelpath = "/home/david/repos/Flux/"
	statedict = torch.load(os.path.join(modelpath, "GazeTR-H-ETH.pt"), 
						   map_location={"cuda:0": "cuda:0"}
						  )
	GazeTR.cuda(); GazeTR.load_state_dict(statedict); GazeTR.eval()
	
	filter_config_2d = {
		'freq': 30,		# system frequency about 30 Hz
		'mincutoff': 0.8,  # value refer to the paper
		'beta': 0.4,	   # value refer to the paper
		'dcutoff': 0.4	 # not mentioned, empirically set
	}
	
	filter_gaze = (OneEuroFilter(**filter_config_2d),
				   OneEuroFilter(**filter_config_2d))
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		
		height, width = frame.shape[:2]
		
		detected_bbox = detector.inference(frame)[0][0]
		bbox = square_box(detected_bbox).astype(int)
		
		img = crop(frame, bbox)
		img = cv2.resize(img, (224, 224))
		
		transform = transforms.ToTensor()
		tensor = transform(img).unsqueeze(0).cuda()

		img = {'face': tensor}
		
		start_time = time.perf_counter()
		gaze = GazeTR(img).detach().cpu().numpy()[0]
		print(time.perf_counter() - start_time)
		print(gazeto3d(gaze))
		bbox_x_center = bbox[0]+((bbox[2]-bbox[0])//2)
		bbox_y_center = bbox[1]+((bbox[3]-bbox[1])//2)
		bbox_width = bbox[2]-bbox[0]
		
		gaze = gazeto3d(gaze)
		
		gaze[0] = filter_gaze[0](gaze[0], time.time())
		gaze[1] = filter_gaze[1](gaze[1], time.time())
		
		cv2.circle(frame, (int(gaze[0]*bbox_width)+bbox_x_center, int(gaze[1]*bbox_width)+bbox_y_center), 15, (125, 255, 0))
		
		cv2.rectangle(frame, (bbox[0], bbox[1]),
							 (bbox[2], bbox[3]), (125, 255, 0), 1)
		
		cv2.imshow('1', frame)
	
		i = i+1
	
		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
