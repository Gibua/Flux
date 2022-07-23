import argparse
import cv2
import numpy as np
import sys, os
import copy
import torch

from src.dataset.dataloader import img_to_tensor, uv_map_to_tensor
from src.dataset.dataloader import make_data_loader, make_dataset, ImageData
from src.model.loss import *
from PIL import Image
from src.util.printer import DecayVarPrinter
from src.visualize.render_mesh import render_face_orthographic, render_uvm
from src.visualize.plot_verts import plot_kpt, compare_kpt
from src.dataset.uv_face import uvm2mesh

from src.run.predict import SADRNv2Predictor

import config

from glue import ULFace
from utils.landmark_utils import *
from utils.face_detection_utils import *

from .Detector import Detector


class Predictor(Detector):
	
	def __init__(self):
		relative_path = './SADRNet/data/saved_model/net_021.pth'
		self.predictor = SADRNv2Predictor(relative_path)
		self.predictor.model.eval()
	
	def pre_process(self, img):
		resized = cv2.resize(img, dsize=(256,256))
		image = (resized / 255.0).astype(np.float32)
		for ii in range(3):
			image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
				image[:, :, ii].var() + 0.001)
		image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
		return image
	
	@torch.no_grad()
	def predict(self, img):
		input_image = self.pre_process(img)
		
		out = self.predictor.model({'img': input_image}, 'predict')
		
		face_uvm_out = out['face_uvm'][0].cpu().permute(1, 2, 0).numpy()
		landmarks = face_uvm_out[uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
		
		return landmarks
		
	def post_process(self, out, bbox):
		bbox_width = bbox[2]-bbox[0]
		
		out *= bbox_width*1.106
		
		landmarks = np.array([(item[0]+bbox[0], item[1]+bbox[1]) for item in out])
		
		return landmarks.copy()

