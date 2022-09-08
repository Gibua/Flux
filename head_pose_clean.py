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

from glue import PFLD_TFLite, ULFace, PIPNet, RetinaFace, PFLD_UltraLight


def putTextCenter(img, text: str, center, fontFace, fontScale: int, color, thickness: int):
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    center_x = np.int(center[0] - (textsize[0]/2.))
    center_y = np.int(center[1] + (textsize[1]/2.))

    cv2.putText(img, text, (center_x, center_y), fontFace, fontScale, color, thickness)


if __name__ == "__main__":
    sys.path.append(os.path.abspath('./modules/Tensorflow2.0-PFLD-'))

    landmark_predictor = PIPNet.Predictor('300W_CELEBA')
    dataset = Datasets.IBUG

    if dataset != Datasets.IBUG:
        mapper = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)

    landmarks_n = landmark_predictor.landmark_count

    face_detector = RetinaFace.Detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera not detected')
    else:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

    cam = Camera(width, height)

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
        ret, frame = cap.read()
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
            r_corner, l_corner = landmark_predictor.get_outer_eye_corners_idx()
            print(landmark_predictor.get_outer_eye_corners_idx())
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

            #print("alo vo",intersect_proportion)

            if (intersect_proportion<0.50):
                is_face_detected, bboxes = face_detector.detect_bbox(frame)
                #print(last_detection)
                if is_face_detected:
                    last_detection = bboxes[0]
                    bbox = last_detection
                    bbox_prev = last_detection
            else:
                bbox_prev = bbox
                bbox = bboxes_average(landmark_bbox, bbox_prev)

        #bbox[0] = max(bbox[0], 0)
        #bbox[1] = max(bbox[1], 0)
        #bbox[2] = min(bbox[2], width-1)
        #bbox[3] = min(bbox[3], height-1)

        if is_face_detected:
            #bbox = face_detector.post_process(bbox)
            bbox = crop_at_corners(bbox, width, height).astype(int)

            img = crop(frame, bbox)

            landmarks = landmark_predictor.predict(frame, bbox)

            xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
            cv2.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), (125, 255, 0), 2)

            #print(landmarks)

            for j in range(landmarks_n):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())

            for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))

            #for point in [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]:
            #for point in mapper.as_list():
            #    cv2.circle(frame,
            #                (np.int32(landmarks[point][0]), np.int32(landmarks[point][1])),
            #                 1, (0, 0, 255))

            #rvec, tvec = hp_estimator.head_pose_from_landmarks(landmarks, calibration=True)

            eye_center = landmarks[landmark_predictor.get_eye_corners_idxs()[0]].mean(axis=0)
            #print(np.linalg.norm(eye_center))
            #print( math.sqrt( (eye_center[0]**2) + (eye_center[1]**2) ) )
            cv2.circle(frame, (np.int32(eye_center[0]), np.int32(eye_center[1])), 3, (255, 150, 0))
            
            pnp_start_time = time.perf_counter()

            if dataset != Datasets.IBUG:
                rvec, tvec = hp_estimator.solve_pose(mapper.map_landmarks(landmarks), True)
            else:
                rvec, tvec = hp_estimator.solve_pose(landmarks, True)

            pnp_count += 1
            pnp_sum += time.perf_counter()-pnp_start_time
            pnp_average = pnp_sum/pnp_count
            #print(pnp_average)

            #for j in range(len(rvec)):
                #t = time.time()
            #    rvec[j] = filter_rvec[j](rvec[j], time.time())

            pitch_color = (210,200,0)
            yaw_color   = (50,150,0)
            roll_color  = (0,0,255)
            cv2.putText(frame, "rvec 1:{:.2f}".format(rvec.flatten()[0]), (0,10+45), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
            cv2.putText(frame, "rvec 2:{:.2f}".format(rvec.flatten()[1]), (0,25+45), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
            cv2.putText(frame, "rvec 3:{:.2f}".format(rvec.flatten()[2]), (0,40+45), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

            hp_estimator.draw_axes(frame, rvec, tvec)
            hp_estimator.draw_angles_text(frame, rvec)
            hp_estimator.draw_annotation_box(frame, rvec, tvec)
            
            #for point in hp_estimator.project_model(rvec, tvec):
            #    cv2.circle(frame, point.astype(int), 3, (0, 233, 255))

            projected = hp_estimator.project_model(rvec, tvec)
            for point in projected:
                cv2.circle(frame, point.astype(int), 3, (0, 233, 255))

            rot_mat = cv2.Rodrigues(rvec)[0]

            rmat = np.matrix(rot_mat)
            tmat = np.matrix(tvec)
            pmat = np.matrix( FaceModel68.LANDMARKS[0].reshape((3,1))*1000, dtype=np.float32)
            cmat = np.matrix(cam.camera_matrix)
            #print(tmat)
            cam_point = rmat * pmat + tmat
            world_point = rmat ** -1 * (cam_point - tmat)

            ext_mat = np.column_stack((rmat,tmat))
            pmat_h = np.row_stack((pmat, [1]))

            #print(cam_point)
            #print(cmat*ext_mat*pmat_h)
            #print(cmat*cam_point)
            poin = cmat*cam_point
            poin[0] /= poin[2]
            poin[1] /= poin[2]
            #print(cmat)
            #print(poin)

            corner_idx = [36, 45]

            ooo = tvec.copy().flatten()+[width/2, height/2, 0]

            #print("/////////", pts_3d[36][:2])
            cv2.circle(frame, (int(poin[0]), int(poin[1])), 5, (255, 100, 255))
            for (x, y, z) in FaceModel68.transform_model(rvec, projected).astype(int):
                cv2.circle(frame, (x, y), 2, (0, 100, 255))

            #print(":::::::::",FaceModel68.transform_model(rvec, projected).astype(int)[0][2])

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
