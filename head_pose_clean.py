from re import L
import sys, os
import time
import math
import cv2
import numpy as np
import copy

from scipy.spatial.transform import Rotation as R

from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets
from common.camera import Camera
from common.head_pose import PoseEstimator2D
from common.face_model_68 import FaceModel68
from modules.OneEuroFilter import OneEuroFilter
from utils.landmark import *
from utils.face_detection import *

from glue import PFLD_TFLite, ULFace, PIPNet, RetinaFace, PFLD_UltraLight, SynergyNet


def projectPoints(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, cmat: np.ndarray, scale = 1):
    rmat = cv2.Rodrigues(rvec)[0]

    projected = ((points.copy()*scale).dot(rmat.T) + tvec.T[0]).dot(cmat.T)
    
    projected[:,0:2] = np.divide(projected[:,:2], projected[:,2:])
    return projected


def draw_axis(img, rvec: np.ndarray, center: np.ndarray, scale = 100):
    axes_points = np.array([[0., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.] ])
    axes_points *= scale

    proj = ((axes_points.dot(rmat.T) + center)).astype(int)

    cv2.line(img, (proj[0][0], proj[0][1]), (proj[1][0],proj[1][1]),(0,0,255),4)
    cv2.line(img, (proj[0][0], proj[0][1]), (proj[2][0],proj[2][1]),(0,255,0),4)
    cv2.line(img, (proj[0][0], proj[0][1]), (proj[3][0],proj[3][1]),(255,0,0),4)

def putTextCenter(img, text: str, center, fontFace, fontScale: int, color, thickness: int):
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    center_x = np.int(center[0] - (textsize[0]/2.))
    center_y = np.int(center[1] + (textsize[1]/2.))

    cv2.putText(img, text, (center_x, center_y), fontFace, fontScale, color, thickness)


if __name__ == "__main__":
    sys.path.append(os.path.abspath('./modules/Tensorflow2.0-PFLD-'))

    landmark_predictor = SynergyNet.Predictor()
    dataset = landmark_predictor.dataset

    if dataset == Datasets.WFLW:
        mapper = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)

    landmarks_n = landmark_predictor.landmark_count

    face_detector = ULFace.Detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera not detected')
    else:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

    f = width
    s_cmat = np.float32([[f, 0, width//2],
                         [0, f, height//2],
                         [0, 0, 1]])
    cam = Camera(width, height, camera_matrix=s_cmat)

    hp_estimator = PoseEstimator2D(cam)

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

    detected_landmarks = np.empty( shape=(0, 0) )
    landmarks = np.empty( shape=(0, 0) )
    bbox = None
    bbox_prev = None
    last_detection = None
    is_face_detected = False

    model3d = FaceModel68()

    rvec_cal = np.zeros((3,1))

    i = 0

    pnp_count = 0
    pnp_sum = 0

    while cap.isOpened():
        ret, original = cap.read()
        frame = original.copy()
        if not ret: break

        height, width = frame.shape[:2]

        is_landmarks_detected = landmarks.size != 0

        if (i == 0) or (i%50 == 0) or (not is_face_detected):
            is_face_detected, bboxes = face_detector.detect_bbox(frame)
            #print(is_face_detected)
            if is_face_detected and (not is_landmarks_detected):
                last_detection = bboxes[0]
                bbox = last_detection
                bbox_prev = last_detection
        if (i != 0) and is_face_detected and is_landmarks_detected:
            r_corner, l_corner = landmark_predictor.get_outer_eye_corners_idxs()
            landmark_bbox = bbox_from_landmark(landmarks, r_corner, l_corner)

            xmin, ymin, xmax, ymax = unwrap_bbox(landmark_bbox.astype(np.int32))
            cv2.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), (0, 0, 255), 2)

            #xmin, ymin, xmax, ymax = unwrap_bbox(last_detection.astype(np.int32))
            #cv2.rectangle(frame, (xmin, ymin),
            #        (xmax, ymax), (0, 200, 255), 2)

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

            xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
            cv2.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), (125, 255, 0), 2)

            for j in range(landmarks_n):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())

            for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))

            
            if dataset not in [Datasets.IBUG, Datasets.AFLW3D]:
                rvec, tvec = hp_estimator.solve_pose(mapper.map_landmarks(landmarks), True)
            else:
                rvec, tvec = hp_estimator.solve_pose(landmarks, True)

            pitch_color = (210,200,0)
            yaw_color   = (50,150,0)
            roll_color  = (0,0,255)
            cv2.putText(frame, "rvec 1:{:.2f}".format(rvec.flatten()[0]), (0,10+45), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
            cv2.putText(frame, "rvec 2:{:.2f}".format(rvec.flatten()[1]), (0,25+45), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
            cv2.putText(frame, "rvec 3:{:.2f}".format(rvec.flatten()[2]), (0,40+45), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)
            
            cx = cam.camera_matrix[0][2]
            cy = cam.camera_matrix[1][2]
            #tvec =  np.concatenate( [landmarks[30]-[cx, cy], [(-4)*translation[2]]]).reshape(3,-1)
            #rvec = head_rot.as_rotvec().reshape((3,1))
            #hp_estimator.draw_axes(frame, rvec, tvec)
            rvec = det_rvec
            hp_estimator.draw_angles_text(frame, rvec)
            hp_estimator.draw_annotation_box(frame, rvec, tvec)
            cam_center = np.array([cx,cy, 0]).reshape(-1,1)
            rmat = cv2.Rodrigues(rvec)[0]
            draw_axis(frame, rvec, det_landmarks[30]*[1,1,-1])#, scale = 10)
            
            #for point in hp_estimator.project_model(rvec, tvec):
            #    cv2.circle(frame, point.astype(int), 3, (0, 233, 255))

            projected = hp_estimator.project_model(rvec, tvec)
            for point in projected:
                cv2.circle(frame, point.astype(int), 3, (0, 233, 255))

            rot_mat = cv2.Rodrigues(rvec)[0]

            #project_3D(rmat, tvec, FaceModel68.LANDMARKS*1000)

            #world_point = rmat ** -1 * (cam_point - tmat)

            #ext_mat = np.column_stack((rmat,tmat))
            #pmat_h = np.row_stack((pmat, [1]))

            corner_idx = [36, 45]

            ooo = tvec.copy().flatten()+[width/2, height/2, 0]

            #t1 = np.matmul(cam.camera_matrix, (R.from_rotvec(tvec.flatten()).apply( FaceModel68.LANDMARKS*1000 )+tvec.reshape((1,3))).T).T
            #proj = projectPoints(FaceModel68.LANDMARKS*1000, rvec, tvec, cam.camera_matrix)

            _, model_trans, _ = cv2.estimateAffine3D(FaceModel68.LANDMARKS*1000, det_landmarks, ransacThreshold=2, confidence=0.99)
            
            #proj = (FaceModel68.LANDMARKS*1000).dot(model_trans[:,:3].T)+model_trans[:,3]
            #for (x, y, z) in proj.astype(int):
            #    cv2.circle(frame, (x, y), 2, (255, 100, 255))

            print(tvec)

            hp_estimator.project_model(rvec, tvec)

        cv2.imshow('1', frame)

        i = i+1

        k = cv2.waitKey(1)
        if k == 27:
            break
        if ((k & 0xFF) == ord('c')) and is_face_detected:
            hp_estimator.set_calibration(landmarks)

    cap.release()
    cv2.destroyAllWindows()
