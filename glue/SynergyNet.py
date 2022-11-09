import cv2
import numpy as np
#import time

from typing import Optional, Tuple

from scipy.spatial.transform import Rotation

from utils.face_detection import square_box, crop
from common.mappings import Datasets

from .LandmarkPredictor import LandmarkPredictor

from modules.SynergyNet.synergy3DMM import SynergyNet
from modules.SynergyNet.utils.inference import predict_sparseVert, predict_pose

import torch

class Predictor(LandmarkPredictor):

    def __init__(self):
        self.__dataset = Datasets.AFLW3D
        self.__landmark_count = 68

        self.__pose_is_provided = True

        self.model = SynergyNet()
    
    @property
    def landmark_count(self):
        return self.__landmark_count

    @property
    def dataset(self):
        return self.__dataset

    @property
    def pose_is_provided(self):
        return self.__pose_is_provided

    
    def _pre_process_bbox(self, bbox, frame_shape: Optional[Tuple[int, int]] = None):
        bbox_scale = 1.2
        #p_bbox = bbox.tolist()
        bbox_c = bbox.copy()
        p_bbox = np.append(bbox_c,1)
        #p_bbox = square_box(p_bbox)
        
        HCenter = (bbox_c[1] + bbox_c[3])/2
        WCenter = (bbox_c[0] + bbox_c[2])/2
        side_len = p_bbox[3]-p_bbox[1]
        margin = (side_len * bbox_scale) // 2
        p_bbox[0], p_bbox[1], p_bbox[2], p_bbox[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin
        
        return p_bbox


    def pre_process(self, img, processed_bbox: np.ndarray):
        raise NotImplementedError


    def predict(self, img, bbox):
        #start_time = time.perf_counter()
        p_bbox = self._pre_process_bbox(bbox, img.shape[:2])

        # enlarge the bbox a little and do a square crop

        input_img = crop(img, p_bbox)
        input_img = cv2.resize(input_img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
        input_img = torch.from_numpy(input_img)
        input_img = input_img.permute(2,0,1)
        input_img = input_img.unsqueeze(0)
        input_img = (input_img - 127.5)/ 128.0

        with torch.no_grad():
            param = self.model.forward_test(input_img)

        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        lmks = predict_sparseVert(param, p_bbox, transform=True)
        pose = predict_pose(param, p_bbox)

        #out = self.model.get_all_outputs(img, p_bbox)
        
        landmarks, head_rotation, translation = self.post_process(lmks, pose)
        #print((time.perf_counter() - start_time))

        return landmarks, head_rotation, translation


    def post_process(self, lmks_3d, pose):
        landmarks = np.column_stack(lmks_3d)
        euler_angles, trans_vec = pose
        rot_vec = Rotation.from_euler('yxz', np.array(euler_angles)*[1,-1,1], degrees= True).as_rotvec()

        return landmarks, rot_vec, trans_vec


    def get_eye_idxs(self):
        right = [36, 37, 38, 39, 40, 41]
        left  = [42, 43, 44, 45, 46, 47]
        return right, left

    def get_eye_corners_idxs(self):
        right = [36, 39]
        left  = [42, 45]
        return right, left

    def get_outer_eye_corners_idxs(self):
        return 36, 45
