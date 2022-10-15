import sys, os
import time
import math
import cv2
import numpy as np
np.set_printoptions(suppress=True)
import copy

import gc


import torch
torch_device = torch.device("cuda:0")
torch.cuda.set_device(torch_device)
torch.cuda.set_per_process_memory_fraction(0.2, device=torch_device )

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
from common.head_pose import PoseEstimator2D, draw_axes, draw_angles_text, draw_annotation_box, ScaledOrthoParameters#, EosHeadPoseEstimator
from common.face_model import ICTFaceModel, ICTFaceModel68, SlothModel

from modules.OneEuroFilter import OneEuroFilter

from utils.landmark import *
from utils.face_detection import *
from utils.mesh import generate_w_mapper

import pickle
import gc

from glue import ULFace, PFLD_UltraLight#, PFLD_TFLite, RetinaFace, SynergyNet  PIPNet






#def pt3d_test(mesh):


def projectPoints(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, cmat: np.ndarray, scale = 1):
    rmat = cv2.Rodrigues(rvec)[0]

    projected = ((points.copy()*scale).dot(rmat.T) + tvec.T[0]).dot(cmat.T)
    
    projected[:,0:2] = np.divide(projected[:,:2], projected[:,2:])
    return projected


def orthoProjection(points: np.ndarray, rvec: np.ndarray, tx, ty, scale):
    rmat = cv2.Rodrigues(rvec)[0]
    translation = np.array([tx, ty, 0])
    projected = ((points.copy()*scale).dot(rmat.T) + translation)

    return projected


def putTextCenter(img, text: str, center, fontFace, fontScale: int, color, thickness: int):
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    center_x = np.int(center[0] - (textsize[0]/2.))
    center_y = np.int(center[1] + (textsize[1]/2.))

    cv2.putText(img, text, (center_x, center_y), fontFace, fontScale, color, thickness)


if __name__ == "__main__":
    #landmark_predictor = PIPNet.Predictor("WFLW")
    landmark_predictor = PFLD_UltraLight.Predictor()
    dataset = landmark_predictor.dataset
     #landmarks_n = landmark_predictor.landmark_count
    landmarks_n = 68

    if dataset == Datasets.WFLW:
        mapper = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)

    face_detector = ULFace.Detector()

    ICT_Model = ICTFaceModel.from_pkl("./common/ICTFaceModel.pkl")
    sloth_model = SlothModel('/home/david/repos/Flux/common/sloth_scaled.glb')

    bs_mapper = generate_w_mapper(ICT_Model.bs_names, sloth_model.bs_name_arr.tolist(), use_jax=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera not detected')
    else:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

    #f = width
    #s_cmat = np.float32([[f, 0, width//2],
    #                     [0, f, height//2],
    #                     [0, 0, 1]])
    cam = PinholeCamera(width, height)#, camera_matrix=s_cmat)

    hp_estimator = PoseEstimator2D(cam)
    fitter = ExpressionFitting(cam.camera_matrix, bs_mapper = bs_mapper)

    #filter_config_2d = {
    #    'freq': 30,        # system frequency about 30 Hz
    #    'mincutoff': 0.8,  # value refer to the paper
    #    'beta': 0.4,       # value refer to the paper
    #    'dcutoff': 0.4     # not mentioned, empirically set
    #}

    #filter_config_2d = {
    #    'freq': 30,        # system frequency about 30 Hz
    #    'mincutoff': 1,  # value refer to the paper
    #    'beta': 0.05,       # value refer to the paper
    #    'dcutoff': 1     # not mentioned, empirically set
    #}

    filter_config_2d = {
        'freq': 30,        # system frequency about 30 Hz
        'mincutoff': 0.8,  # value refer to the paper
        'beta': 0.4,       # value refer to the paper
        'dcutoff': 0.4     # not mentioned, empirically set
    }

    filter_2d = [(OneEuroFilter(**filter_config_2d),
                  OneEuroFilter(**filter_config_2d))
                  for _ in range(landmarks_n)]

    filter_rvec = (OneEuroFilter(**filter_config_2d),
                   OneEuroFilter(**filter_config_2d),
                   OneEuroFilter(**filter_config_2d))
    
    landmarks = np.empty( shape=(0, 0) )
    bbox = None
    bbox_prev = None
    last_detection = None
    is_face_detected = False
    w = None
    n_its = 0
    last_time = time.time()
    model_img = np.full((height, width, 3), 64, dtype=np.uint8)

    while cap.isOpened():
        ret, original = cap.read()
        frame = original.copy()
        if not ret: break

        height, width = frame.shape[:2]

        model_img = np.full((height, width, 3), 64, dtype=np.uint8)

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
            #bbox = face_detector.post_process(bbox)
            bbox = crop_at_corners(bbox, width, height).astype(int)

            img = crop(frame, bbox)

            if landmark_predictor.pose_is_provided:
                det_landmarks, det_rvec, det_tvec = landmark_predictor.predict(frame, bbox)
            else:
                det_landmarks = landmark_predictor.predict(frame, bbox)
            landmarks = det_landmarks[:,0:2]

            if dataset not in [Datasets.IBUG, Datasets.AFLW3D]:
                landmarks = mapper.map_landmarks(landmarks)
            
            xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
            cv2.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), (125, 255, 0), 2)
            
            for j in range(landmarks_n):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())
            
            for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
            
            rvec, tvec = hp_estimator.solve_pose(landmarks, True)
            
            #pitch_color = (210,200,0)
            #yaw_color   = (50,150,0)
            #roll_color  = (0,0,255)
            #cv2.putText(frame, "rvec 1:{:.2f}".format(rvec.flatten()[0]), (0,10+45), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
            #cv2.putText(frame, "rvec 2:{:.2f}".format(rvec.flatten()[1]), (0,25+45), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
            #cv2.putText(frame, "rvec 3:{:.2f}".format(rvec.flatten()[2]), (0,40+45), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)
            
            cx = cam.camera_matrix[0][2]
            cy = cam.camera_matrix[1][2]
            #tvec =  np.concatenate( [landmarks[30]-[cx, cy], [(-4)*translation[2]]]).reshape(3,-1)
            #rvec = head_rot.as_rotvec().reshape((3,1))
            #hp_estimator.draw_axes(frame, rvec, tvec)
            #rvec = det_rvec
            
            draw_angles_text(frame, rvec)
            #draw_annotation_box(frame, rvec, tvec, cam)
            draw_axes(frame, rvec, tvec, cam.camera_matrix)#, scale = 1000)

            #projected = hp_estimator.project_model(rvec, tvec)
            #xy_center = landmarks[30]
            #projected = orthoProjection(hp_estimator.model_points_68, rvec, xy_center[0], xy_center[1], cam.get_focal()/tvec[2][0])
            
            w_dlp = fitter.fit(landmarks, rvec, tvec, method = 'scipy_lm')
            w_tensor = torch.from_dlpack(w_dlp).float()
            
            #weighted = ICT_Model.apply_weights(w)
            
            mesh_weighted = sloth_model.apply_weights_to_mesh(w_tensor)
            
            weighted = mesh_weighted.verts_packed().cpu().numpy()
            faces =  mesh_weighted.faces_list()[0].cpu().numpy()
            projected = projectPoints(weighted*[1,-1,1], rvec, tvec, cam.camera_matrix, scale=15)
            #scale = cam.camera_matrix[0][0]/tvec.ravel()[2]
            #c_x = cam.camera_matrix[0][2]
            #c_y = cam.camera_matrix[1][2]
            #t_x = c_x + tvec.ravel()[0]*scale
            #t_y = c_y + tvec.ravel()[1]*scale
            #projected = orthoProjection(weighted, rvec, t_x, t_y, scale)
            
            #try:
            #    for point in projected.astype(np.int32):
            #        cv2.circle(frame, (point[0], point[1]), 1, (200, 200, 200))
            #    a = False
            #except:
            #    pass
            cv2.polylines(model_img, projected[:,:2][faces].astype(int), True, (255, 255, 255), 1, cv2.LINE_AA)
            #frame = render(frame, [projected.T.astype(np.float32)], ICT_Model['topology'].astype(np.int32), alpha=0.7)
            #for tri in ICT_Model['topology']:
            #    cv2.line(frame, projected[tri[0]][:2].astype(int), projected[tri[1]][:2].astype(int), (0,255,0),1)
                #cv2.line(frame, projected[tri[0]][:2].astype(int), projected[tri[2]][:2].astype(int), (0,255,0),1)
                #cv2.line(frame, projected[tri[1]][:2].astype(int), projected[tri[2]][:2].astype(int), (0,255,0),1)


            rot_mat = cv2.Rodrigues(rvec)[0]

            hp_estimator.project_model(rvec, tvec)

        cv2.imshow('features', frame)
        cv2.imshow('retarget', model_img)

        n_its += 1

        k = cv2.waitKey(1)
        if k == 27:
            break
        if ((k & 0xFF) == ord('c')) and is_face_detected:
            hp_estimator.set_calibration(landmarks)
            #hp_estimator.set_calibration(rvec)

    cap.release()
    cv2.destroyAllWindows()
