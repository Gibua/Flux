import sys, os
import time
import math
import cv2
import numpy as np
np.set_printoptions(suppress=True)
import copy

from decimal import Decimal as D

import gc

from scipy.spatial.transform import Rotation as R

from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets
from common.camera import PinholeCamera
from common.fitting import ExpressionFitting
from common.head_pose import PoseEstimator2D, draw_axes, draw_angles_text, draw_annotation_box, ScaledOrthoParameters#, EosHeadPoseEstimator
from common.face_model import ICTFaceModel, ICTFaceModel68, SlothModel

from modules.OneEuroFilter import OneEuroFilter

from utils.landmark import *
from utils.face_detection import *
from utils.mesh import generate_w_mapper

import pickle
import gc

from glue import ULFace, PFLD_UltraLight#, PFLD_TFLite, RetinaFace, SynergyNet  PIPNet



import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (FoVPerspectiveCameras,
                                PointLights,
                                RasterizationSettings,
                                MeshRenderer,
                                MeshRasterizer,
                                HardPhongShader)

def calibrate(face_detector, landmark_predictor, hp_estimator): 
    dataset = landmark_predictor.dataset
    landmarks_n = 68

    if dataset == Datasets.WFLW:
        mapper = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)

    model_3d = ICTFaceModel68.from_pkl("./common/ICTFaceModel.pkl", load_blendshapes=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera not detected')
        return
    else:
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        
    cam = PinholeCamera(width, height)

    hp_estimator = PoseEstimator2D(cam)

    filter_config = {
        'freq': 30,        # Values from: https://github.com/XinArkh/VNect/ ([Mehta et al. 2017])
        'mincutoff': 0.8,  # Left the same because those values give good results, empirically
        'beta': 0.4,       # 
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
    w = None
    n_its = 0
    last_time = time.time()

    while cap.isOpened():
        ret, original = cap.read()
        frame = original.copy()
        if not ret: break

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

            xmin, ymin, xmax, ymax = unwrap_bbox(landmark_bbox.astype(np.int32))
            cv2.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), (0, 0, 255), 2)

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
            bbox = crop_at_corners(bbox, width, height).astype(int)

            if landmark_predictor.pose_is_provided:
                det_landmarks, det_rvec, det_tvec = landmark_predictor.predict(frame, bbox)
            else:
                det_landmarks = landmark_predictor.predict(frame, bbox)
            landmarks = det_landmarks[:,0:2]

            if dataset not in [Datasets.IBUG, Datasets.AFLW3D]:
                landmarks = mapper.map_landmarks(landmarks)
            
            for j in range(landmarks_n):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())
            
            for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
            
            rvec, tvec = hp_estimator.solve_pose(landmarks, True)
            
            w_dlp, w_np = fitter.fit(landmarks, rvec, tvec, float(b_fit), float(b_prior), float(b_sparse), method = 'jaxopt_lm', debug=True)
            w_tensor = torch.from_dlpack(w_dlp).float()
            
            ict_weighted = ICT_Model.apply_weights(w_np)
            
            mesh_weighted = sloth_model.apply_weights_to_mesh(w_tensor)

            model_img = pt3d_test(mesh_weighted, rvec = None, tvec = None)

            projected = projectPoints(ict_weighted, rvec, tvec, cam.camera_matrix)#, scale=15)
            
            cv2.polylines(frame, projected[:,:2][ICT_Model.faces].astype(int), True, (255, 255, 255), 1, cv2.LINE_AA)

            rot_mat = cv2.Rodrigues(rvec)[0]

            hp_estimator.project_model(rvec, tvec)

        cv2.imshow('features', frame)
        cv2.imshow('retarget', model_img)

        n_its += 1

        k = cv2.waitKey(1)
        max_b = '5'
        if k == 27:
            break
        if ((k & 0xFF) == ord('c')) and is_face_detected:
            hp_estimator.set_calibration(landmarks)
            #hp_estimator.set_calibration(rvec)


    cap.release()
    cv2.destroyAllWindows()
