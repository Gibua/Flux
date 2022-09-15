import cv2
import numpy as np

from typing import Optional, Tuple

from scipy.spatial.transform import Rotation

from utils.face_detection import square_box, crop
from common.mappings import Datasets

from .LandmarkPredictor import LandmarkPredictor

from modules.SynergyNet.synergy3DMM import SynergyNet


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
        bbox_scale = 1
        p_bbox = bbox.tolist()
        p_bbox.append(1)
        #p_bbox = square_box(p_bbox)

        return [p_bbox]


    def pre_process(self, img, processed_bbox: np.ndarray):
        raise NotImplementedError


    def predict(self, img, bbox):
        p_bbox = self._pre_process_bbox(bbox, img.shape[:2])
        
        out = self.model.get_all_outputs(img, p_bbox)
        
        landmarks, head_rotation, translation = self.post_process(out)

        return landmarks, head_rotation, translation


    def post_process(self, preds):
        lmk3d, mesh, pose = preds

        landmarks = np.column_stack(lmk3d[0])
        euler_angles, trans_vec = pose[0]
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
