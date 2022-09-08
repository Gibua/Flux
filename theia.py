import cv2
import numpy as np

import copy
from scipy.spatial.transform import Rotation as R

from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets

from utils.camera import Camera

import pytheia as pt

def draw_euler_axis(image, axis_pts, euler_angles):
    """
        draw euler axes in the image center
    """
    center = (image.shape[1]//2, image.shape[0]//2)

    axis_pts = axis_pts.astype(np.int32)
    pitch_point = tuple(axis_pts[0].ravel())
    yaw_point   = tuple(axis_pts[1].ravel())
    roll_point  = tuple(axis_pts[2].ravel())

    pitch_color = (255,255,0)
    yaw_color   = (0,255,0)
    roll_color  = (0,0,255)

    pitch, yaw, roll = euler_angles

    cv2.line(image, center,  pitch_point, pitch_color, 2)
    cv2.line(image, center,  yaw_point, yaw_color, 2)
    cv2.line(image, center,  roll_point, roll_color, 2)
    cv2.putText(image, "Pitch:{:.2f}".format(pitch), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
    cv2.putText(image, "Yaw:{:.2f}".format(yaw), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
    cv2.putText(image, "Roll:{:.2f}".format(roll), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

    # origin
    cv2.circle(image, center, 2, (255,255,255), -1)
    return image


def draw_euler_angles(image, rvec, tvec, euler_angles, intrinsic_matrix):
    # i, j, k axes in world 3D coord.
    axis = np.identity(3) * 5
    # axis_img_pts = intrinsic * exstrinsic * axis
    axis_pts = cv2.projectPoints(axis, rvec, tvec, intrinsic_matrix, None)[0]
    image = draw_euler_axis(image, axis_pts, euler_angles)

    return image


def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 10
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 10
    front_depth = 20
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    # Map to 2d image points
    #(point_2d, _) = cv2.projectPoints(point_3d,
                                      #rotation_vector,
                                      #translation_vector,
                                      #camera_matrix,
    #                                  np.zeros((4, 1)))
    #point_2d = np.int32(point_2d.reshape(-1, 2))
    point_3d = 10*(R.from_rotvec(rotation_vector.reshape((3,))).apply(point_3d)+15)
    point_2d = np.delete(point_3d, 2, axis=1).astype(np.int32)
    print(point_2d+50)

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, (255,0,0), line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    #xyz = cv2.RQDecomp3x3(cv2.Rodrigues(rotation_vector)[0])[0]
    xyz = R.from_rotvec(rotation_vector.flatten()).as_euler('xyz', degrees=True)

    pitch_color = (210,200,0)
    yaw_color   = (50,150,0)
    roll_color  = (0,0,255)

    cv2.putText(image, "Pitch:{:.2f}".format(xyz[0]), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
    cv2.putText(image, "Yaw:{:.2f}".format(xyz[1]), (0,25), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
    cv2.putText(image, "Roll:{:.2f}".format(xyz[2]), (0,40), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)


def annotated_frame(frame, pupil_right_coords, pupil_left_coords):
    """Returns the main frame with pupils highlighted"""
    frame = frame.copy()

    color = (0, 255, 0)
    x_left, y_left = pupil_left_coords
    x_right, y_right = pupil_right_coords
    cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
    cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
    cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
    cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

    return frame


def putTextCenter(img, text: str, center, fontFace, fontScale: int, color, thickness: int):
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    center_x = np.int(center[0] - (textsize[0]/2.))
    center_y = np.int(center[1] + (textsize[1]/2.))

    cv2.putText(img, text, (center_x, center_y), fontFace, fontScale, color, thickness)


if __name__ == "__main__":
    import sys, os
    import time
    sys.path.append(os.path.abspath('./modules/Tensorflow2.0-PFLD-'))
    from glue import PFLD_TFLite, ULFace
    from utils.landmark import *
    from utils.face_detection import *
    import math
    from modules.OneEuroFilter import OneEuroFilter
    from common.face_model_68 import FaceModel68
    import inspectshow

    show = inspectshow.tree()
    show(pytheia)

    landmark_predictor = PFLD_TFLite.Predictor()
    face_detector = ULFace.Detector()


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera not detected')
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    cap_camera = Camera(width, height)


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
                  for _ in range(98)]

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

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret: break

        height, width = frame.shape[:2]

        is_landmarks_detected = landmarks.size != 0

        if (i == 0) or (i%50 == 0):
            is_face_detected, last_detection = face_detector.detect_bbox(frame)
            #print(is_face_detected)
            if is_face_detected and (not is_landmarks_detected):
                bbox = last_detection.copy()
                bbox_prev = last_detection
        if (i != 0) and is_face_detected and is_landmarks_detected:
            landmark_bbox = bbox_from_landmark(landmarks)

            intersection = bbox_intersect(last_detection, landmark_bbox)

            landmark_bbox_area = bbox_area(np.array(landmark_bbox))
            intersect_area = bbox_area(intersection)
            intersect_proportion = intersect_area/landmark_bbox_area

            #print(intersect_proportion)

            if (intersect_proportion<0.65):
                is_face_detected, last_detection = face_detector.detect_bbox(frame)
                #print(last_detection)
                if is_face_detected:
                    bbox = last_detection.copy()
                    bbox_prev = last_detection
            else:
                bbox_prev = bbox
                bbox = bboxes_average(landmark_bbox, bbox_prev)

        if is_face_detected:
            bbox = face_detector.post_process(bbox)

            xmin, ymin, xmax, ymax = unwrap_bbox(bbox)

            cv2.rectangle(frame, (xmin, ymin),
                    (xmax, ymax), (125, 255, 0), 2)

            img = crop(frame, bbox)

            detected_landmarks = landmark_predictor.predict(img)

            landmarks = landmark_predictor.post_process(detected_landmarks, bbox)

            start_time = time.time()
            for j in range(98):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())
            #print(time.time() - start_time)

            for (x, y) in landmarks:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))

            #for point in [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]:
            for point in LandmarkMapper(Datasets.WFLW, Datasets.IBUG).as_list():
                cv2.circle(frame,
                            (np.int32(landmarks[point][0]), np.int32(landmarks[point][1])),
                             1, (0, 0, 255))

            #print(pt.sfm.DlsPnp(landmarks, FaceModel68.LANDMARKS))

            #draw_annotation_box(frame, rvec, tvec, cap_camera.camera_matrix, color=(0, 255, 0))

        cv2.imshow('1', frame)

        i = i+1

        k = cv2.waitKey(1)
        if k == 27:
            break
        if ((k & 0xFF) == ord('c')) and is_face_detected:
            hp_estimator.set_calibration(landmarks)

    cap.release()
    cv2.destroyAllWindows()
