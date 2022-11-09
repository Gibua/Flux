import sys, os
import time
import math
import cv2
import numpy as np
np.set_printoptions(suppress=True)
import copy

from decimal import Decimal as D

import gc


import torch
torch_device = torch.device("cuda:0")
torch.cuda.set_device(torch_device)
torch.cuda.set_per_process_memory_fraction(0.2, device=torch_device)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
#tf.config.experimental.set_visible_devices([], "GPU")

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'

from scipy.spatial.transform import Rotation as R

from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets
from common.camera import PinholeCamera
from common.fitting import ExpressionFitting
from common.head_pose import PoseEstimator2D, draw_axes, draw_angles_text, draw_annotation_box
from common.face_model import ICTFaceModel68, SlothModel, ICTModelPT3D

from modules.OneEuroFilter import OneEuroFilter

from utils.landmark import *
from utils.face_detection import *
from utils.mesh import generate_w_mapper

from VideoGet import VideoGet
import gc

from glue import RetinaFace, PFLD_UltraLight, PFLD_TFLite, ULFace, RetinaFace, SynergyNet, PIPNet



import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PointLights,
                                RasterizationSettings,
                                MeshRenderer,
                                MeshRasterizer,
                                HardPhongShader)
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras


def orthoProjection(points: np.ndarray, rvec: np.ndarray, tx, ty, scale):
    rmat = cv2.Rodrigues(rvec)[0]
    translation = np.array([tx, ty, 0])
    projected = ((points.copy()*scale).dot(rmat.T) + translation)

    return projected


def putTextCenter(img, text: str, center, fontFace, fontScale: int, color, thickness: int):
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    center_x = np.int32(center[0] - (textsize[0]/2.))
    center_y = np.int32(center[1] + (textsize[1]/2.))

    cv2.putText(img, text, (center_x, center_y), fontFace, fontScale, color, thickness)


#if __name__ == "__main__":
def main(video_path, face_detector, landmark_predictor):
    cap = cv2.VideoCapture(video_path)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #landmark_predictor = PIPNet.Predictor("WFLW")
    
    dataset = landmark_predictor.dataset
    #landmarks_n = landmark_predictor.landmark_count
    landmarks_n = 68
    if dataset == Datasets.WFLW:
        mapper = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)

    cam = PinholeCamera(width, height)#, camera_matrix=s_cmat)

    hp_estimator = PoseEstimator2D(cam)

    #filter_config_2d = {
    #    'freq': 30,
    #    'mincutoff': 0.8,
    #    'beta': 0.4,
    #    'dcutoff': 0.4 
    #}

    #filter_config_2d = {
    #    'freq': 30,
    #    'mincutoff': 1,
    #    'beta': 0.05,
    #    'dcutoff': 1
    #}

    filter_config = {
        'freq': 30,        # Values from: https://github.com/XinArkh/VNect/ ([Mehta et al. 2017])
        'mincutoff': 0.4,  # Left the same because those values give good results, empirically
        'beta': 0.6,       # 
        'dcutoff': 0.4     # 
    }
    
    filter_2d = [(OneEuroFilter(**filter_config),
                  OneEuroFilter(**filter_config))
                  for _ in range(landmarks_n)]

    filter_rvec = (OneEuroFilter(**filter_config),
                   OneEuroFilter(**filter_config),
                   OneEuroFilter(**filter_config))
    
    landmarks = np.empty( shape=(0, 0) )
    bbox = None
    bbox_prev = None
    last_detection = None
    is_face_detected = False
    n_its = 0
    last_time = time.time()
    
    while cap.isOpened():
        
        ret, original = cap.read()
        if not ret: break
        frame = original.copy()
        
        height, width = frame.shape[:2]

        is_landmarks_detected = landmarks.size != 0
        
        time_elapsed = time.time()-last_time
        if (n_its == 0) or (time_elapsed > 2.5) or (not is_face_detected):
            last_time = time.time()
            is_face_detected, bboxes = face_detector.detect_bbox(frame)
            if is_face_detected and (not is_landmarks_detected):
                last_detection = bboxes[0]
                bbox = last_detection
                bbox_prev = last_detection
        if (n_its > 0) and is_face_detected and is_landmarks_detected:
            r_corner, l_corner = (45, 36)
            landmark_bbox = bbox_from_landmark(landmarks, r_corner, l_corner)

            intersection = bbox_intersect(last_detection, landmark_bbox)

            landmark_bbox_area = bbox_area(landmark_bbox)
            last_detection_area = bbox_area(last_detection)
            intersect_area = bbox_area(intersection)
            intersect_proportion = intersect_area/max(landmark_bbox_area,last_detection_area)

            if (intersect_proportion<0.50):
                is_face_detected, bboxes = face_detector.detect_bbox(frame)
                if is_face_detected:
                    last_detection = bboxes[0]
                    bbox = last_detection
                    bbox_prev = last_detection
            else:
                bbox_prev = bbox
                bbox = bboxes_average(landmark_bbox, bbox_prev)
        
        if is_face_detected:
            
            #bbox = face_detector.post_process(bbox)
            bbox = crop_at_corners(bbox, width, height).astype(int)

            if landmark_predictor.pose_is_provided:
                det_landmarks, det_rvec, det_tvec = landmark_predictor.predict(frame, bbox)
            else:
                det_landmarks = landmark_predictor.predict(frame, bbox)
            landmarks = det_landmarks[:,0:2]

            if dataset not in [Datasets.IBUG, Datasets.AFLW3D]:
                landmarks = mapper.map_landmarks(landmarks)
            
            #xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
            #cv2.rectangle(frame, (xmin, ymin),
            #                     (xmax, ymax), (125, 255, 0), 2)
            
            for j in range(landmarks_n):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())
            
            for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))

            rvec, tvec = hp_estimator.solve_pose(landmarks, True)

            print(R.from_rotvec(rvec.ravel()).as_euler('xyz', degrees=True))

            draw_axes(frame, rvec, tvec, cam.camera_matrix)
        
        cv2.imshow('features', frame)
        
        n_its += 1
        
        k = cv2.waitKey()
        
        if k == 27:
            break
        if ((k & 0xFF) == ord('c')) and is_face_detected:
            hp_estimator.set_calibration(landmarks)
            #hp_estimator.set_calibration(rvec)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import traceback

    video_path = '/home/david/Downloads/jam1.avi'
    #video_path = 0
    landmark_predictor = PFLD_UltraLight.Predictor()
    face_detector = ULFace.Detector()
    try:
        main(video_path, face_detector, landmark_predictor)
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        #print(str(type(e).__name__)+':',e)
        traceback.print_exc()
    
    sys.exit()