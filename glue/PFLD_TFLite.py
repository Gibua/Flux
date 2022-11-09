import os
import cv2
import numpy as np
import tensorflow as tf

from scipy.special import softmax, expit
from typing import Optional, Tuple

from modules.TFLitePFLD.Model.utils import parse_arguments, Normalization, color_
from modules.TFLitePFLD.Model.datasets import DateSet

from .LandmarkPredictor import LandmarkPredictor

from utils.face_detection import square_box, crop
from common.mappings import Datasets

import time


class Predictor(LandmarkPredictor):


    def __init__(self):
        relative_path = './modules/TFLitePFLD/models/tflite/pfld_infer.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=os.path.abspath(relative_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.__dataset = Datasets.WFLW
        self.__landmark_count = 98

        self.__pose_is_provided = False


    @property
    def landmark_count(self):
        return self.__landmark_count

    @property
    def dataset(self):
        return self.__dataset

    @property
    def pose_is_provided(self):
        return self.__pose_is_provided

    def set_input_tensor(self, interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(tensor_index, image)

    def _pre_process_bbox(self, bbox, frame_shape: Optional[Tuple[int, int]] = None):
        bbox_scale = 1

        p_bbox = bbox.copy()

        p_bbox = square_box(p_bbox)

        return p_bbox.astype(int)

    def pre_process(self, img, bbox):        
        cropped = crop(img, bbox)

        imgrgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(imgrgb, dsize=(112,112))
        image_rgb = resized[..., ::-1]
        image_norm = image_rgb.astype(np.float32)
        cv2.normalize(image_norm, image_norm,
            alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
        return image_norm[None, ...]

    def predict(self, img, bbox):
        p_bbox = self._pre_process_bbox(bbox, img.shape[:2])
        input_tensor = self.pre_process(img, p_bbox)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(self.output_details[0]['index'])
        preds = expit(preds)
        preds = preds.reshape(-1, 2)
            
        landmarks = self.post_process(preds, p_bbox)

        return landmarks

    def post_process(self, landmarks, bbox):
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]
        processed = (landmarks.copy()*[bbox_width, bbox_height])+[bbox[0], bbox[1]]
        return processed

    def get_eye_idxs(self):
        right = [60, 61, 62, 63, 64, 65, 66, 67]
        left  = [68, 69, 70, 71, 72, 73, 74, 75]
        return right, left

    def get_eye_corners_idxs(self):
        right = [60, 64]
        left  = [68, 72]
        return right, left

    def get_outer_eye_corners_idxs(self): #  right-to-left
        return 60, 72