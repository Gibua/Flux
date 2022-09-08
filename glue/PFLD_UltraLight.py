import argparse
import cv2
import numpy as np
import sys, os
import copy

from ncnn_vulkan import ncnn

from typing import Optional, Tuple

from utils.face_detection import square_box, crop

from .LandmarkPredictor import LandmarkPredictor


class Predictor(LandmarkPredictor):

    def __init__(self):

        self._landmark_count = 98
        
        self.model = ncnn.Net()

        self.g_blob_pool_allocator = ncnn.UnlockedPoolAllocator()
        self.g_workspace_pool_allocator = ncnn.PoolAllocator()
        self.g_blob_pool_allocator.clear()
        self.g_workspace_pool_allocator.clear()

        param_path = "./weights/PFLD_Ghost_1_112/PFLD_Ghost_112_1.param"
        bin_path = "./weights/PFLD_Ghost_1_112/PFLD_Ghost_112_1.bin"

        ret = self.model.load_param(param_path)
        self.model.load_model(bin_path)

        self.num_of_threads = ncnn.get_cpu_count()
    
    @property
    def landmark_count(self):
        return self._landmark_count

    
    def _pre_process_bbox(self, bbox, frame_shape: Optional[Tuple[int, int]] = None):
        bbox_scale = 1
        p_bbox = bbox.copy()
        p_bbox = square_box(p_bbox)

        return p_bbox.astype(int)

    def pre_process(self, img, processed_bbox: np.ndarray):
        cropped = crop(img, processed_bbox).copy()
        
        #print(cropped.shape)
        #while True:
        #    cv2.imshow("cropped", cropped)
        
        crop_height, crop_width = cropped.shape[:2]

        mat_in = ncnn.Mat.from_pixels_resize(cropped,
                                ncnn.Mat.PixelType.PIXEL_BGR,
                                crop_width, crop_height,
                                112, 112)
        mean_vals = np.float32([])
        norm_vals = np.float32([1 / 255.0, 1 / 255.0, 1 / 255.0])
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        return mat_in


    def predict(self, img, bbox):
        p_bbox = self._pre_process_bbox(bbox, img.shape[:2])
        mat_in = self.pre_process(img, p_bbox)

        ex = self.model.create_extractor()
        ex.set_num_threads(self.num_of_threads)
       
        # Make sure the input and output names match the param file
        ex.input("input", mat_in)
        #start_time = time.perf_counter()
        ret, mat_out = ex.extract("output")
        print("yesyesyesyesyesyesyesyesyesyes")
        #print(time.perf_counter() - start_time)
        out = np.array(mat_out.clone())

        # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
        #output = out.transpose(1, 2, 0) * 255

        # Save image using opencv
        #cv2.imwrite('./out.png', output)
            
        landmarks = self.post_process(out, p_bbox)

        return landmarks

    def post_process(self, preds, bbox):
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]
        landmarks = np.column_stack([preds[0::2], preds[1::2]])
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

    def get_outer_eye_corners_idxs(self):
        return 60, 72