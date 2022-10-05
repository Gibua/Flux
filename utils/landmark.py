import cv2
import numpy as np
import os
import sys
import copy

from common.face_model_68 import FaceModel68


def inter_ocular_dist(landmarks):
    r_eye_r_corner = landmarks[36]
    l_eye_l_corner = landmarks[45]
    return np.linalg.norm(l_eye_l_corner-r_eye_r_corner)


def point_origin_dist(point):
    return np.linalg.norm(point)


def face_center(landmarks):
    nose_base_i = FaceModel68.NOSE_INDICES
    r_eye_i = FaceModel68.REYE_INDICES
    l_eye_i = FaceModel68.LEYE_INDICES

    center = landmarks[np.concatenate([r_eye_i, l_eye_i, nose_base_i])].mean(axis=0)

    return center

def eye_centers(landmarks):
    r_eye_i = FaceModel68.REYE_INDICES
    l_eye_i = FaceModel68.LEYE_INDICES

    left_center  = landmarks[l_eye_i].mean(axis=0)
    right_center = landmarks[r_eye_i].mean(axis=0)

    return right_center, left_center
