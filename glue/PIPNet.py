import cv2
import numpy as np
import os
import copy
import importlib

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from scipy.special import softmax, expit
from PIL import Image

from modules.PIPNet.networks import *
from modules.PIPNet.functions import *

from utils.face_detection import square_box, crop, crop_at_corners

from .LandmarkPredictor import LandmarkPredictor

from typing import Optional, Tuple

class Predictor(LandmarkPredictor):


    def __init__(self, weight: str):
        
        if weight == 'WFLW':
            data_name = "WFLW"
            experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10"
            self._landmark_count = 98

            self.get_eye_idxs = self._wflw_eye_idxs
            self.get_eye_corners_idxs = self._wflw_eye_corners_idxs
            self.get_outer_eye_corners_idx = self._wflw_outer_eye_corners_idxs

        else:
            if weight == '300W_CELEBA':
                data_name = "data_300W_CELEBA"
                experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10_wcc"
                self._landmark_count = 68
            elif weight == '300W_COFW_WFLW':
                data_name = "data_300W_COFW_WFLW"
                experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10_wcc"
                self._landmark_count = 68
            
            self.get_eye_idxs = self._ibug_eye_idxs
            self.get_eye_corners_idxs = self._ibug_eye_corners_idxs
            self.get_outer_eye_corners_idx = self._ibug_outer_eye_corners_idxs
        
        config_path = 'modules.PIPNet.experiments.{}.{}'.format(data_name, experiment_name)
        weight_path = os.path.join('./weights/PIPNet', data_name, experiment_name)

        my_config = importlib.import_module(config_path, package='PIPNetConfig')
        Config = getattr(my_config, 'Config')
        self.cfg = Config()
        self.cfg.experiment_name = experiment_name
        self.cfg.data_name = data_name
        
        self.weight_file = os.path.join(weight_path, 'epoch%d.pth' % (self.cfg.num_epochs-1))
        self.meanface_indices, self.reverse_index1, self.reverse_index2, self.max_len = get_meanface(os.path.join('./modules/PIPNet/data', self.cfg.data_name, 'meanface.txt'), self.cfg.num_nb)
        
        if self.cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=self.cfg.pretrained)
            net = Pip_resnet18(resnet18, self.cfg.num_nb, num_lms=self.cfg.num_lms, input_size=self.cfg.input_size, net_stride=self.cfg.net_stride)
        else:
            print('No such backbone!')
        
        if self.cfg.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        net = net.to(self.device)
        
        state_dict = torch.load(self.weight_file, map_location=self.device)
        net.load_state_dict(state_dict)
        net.eval()

        self.net = net

        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225])
        self._preprocess = transforms.Compose([transforms.Resize((self.cfg.input_size, self.cfg.input_size)), transforms.ToTensor(), self._normalize])


    @property
    def landmark_count(self):
        return self._landmark_count
        

    def _pre_process_bbox(self, bbox, frame_shape: Optional[Tuple[int, int]] = None):
        bbox_scale = 1.2

        det_bbox = bbox.copy()

        det_width = det_bbox[2] - det_bbox[0] + 1
        det_height = det_bbox[3] - det_bbox[1] + 1

        det_bbox[0] -= int(det_width  * (bbox_scale-1)/2)
        det_bbox[1] += int(det_height * (bbox_scale-1)/2)
        det_bbox[2] += int(det_width  * (bbox_scale-1)/2)
        det_bbox[3] += int(det_height * (bbox_scale-1)/2)

        if frame_shape is not None:
            frame_height = frame_shape[0]
            frame_width  = frame_shape[1]
            det_bbox[0] = max(det_bbox[0], 0)
            det_bbox[1] = max(det_bbox[1], 0)
            det_bbox[2] = min(det_bbox[2], frame_width-1)
            det_bbox[3] = min(det_bbox[3], frame_height-1)

        return det_bbox


    def pre_process(self, img, bbox):
        det_bbox = self._pre_process_bbox(bbox, img.shape[:2])

        cropped = crop(img, det_bbox)
        
        resized = cv2.resize(cropped, (self.cfg.input_size, self.cfg.input_size))
        inputs = Image.fromarray(resized[:,:,::-1].astype('uint8'), 'RGB')
        inputs = self._preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(self.device)

        return inputs
    

    def predict(self, img, bbox):
        inputs = self.pre_process(img, bbox)
        #inputs = inputs.to(torch.device("cpu"))
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(self.net, inputs, self._preprocess, 
                                                                                                 self.cfg.input_size, self.cfg.net_stride, 
                                                                                                 self.cfg.num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        preds = lms_pred_merge.cpu().numpy()

        landmarks = self.post_process(preds, bbox)

        return landmarks


    def post_process(self, preds, bbox):
        processed = preds.copy()
        proc_bbox = self._pre_process_bbox(bbox)
        bbox_width  = proc_bbox[2]-proc_bbox[0]
        bbox_height = proc_bbox[3]-proc_bbox[1]
        processed = np.array([[processed[i*2] * bbox_width + proc_bbox[0],
                               processed[i*2+1] * bbox_height + proc_bbox[1]] 
                               for i in range(self.cfg.num_lms)])
        return processed

    def _ibug_eye_idxs(self):
        right = [36, 37, 38, 39, 40, 41]
        left  = [42, 43, 44, 45, 46, 47]
        return right, left

    def _ibug_eye_corners_idxs(self):
        right = [36, 39]
        left  = [42, 45]
        return right, left

    def _ibug_outer_eye_corners_idxs(self):
        return 36, 45


    def _wflw_eye_idxs(self):
        right = [60, 61, 62, 63, 64, 65, 66, 67]
        left  = [68, 69, 70, 71, 72, 73, 74, 75]
        return right, left

    def _wflw_eye_corners_idxs(self):
        right = [60, 64]
        left  = [68, 72]
        return right, left

    def _wflw_outer_eye_corners_idxs(self):
        return 60, 72

