from itertools import starmap
import numpy as np
from math import sqrt, dist, atan2, cos, sin, pi

import time

def crop(frame, bbox):    
    xmin = max(bbox[0], 0)
    ymin = max(bbox[1], 0)
    xmax = min(bbox[2], frame.shape[1]-1)
    ymax = min(bbox[3], frame.shape[0]-1)
    cut = frame[ymin:ymax, xmin:xmax,:].copy()
	
    return cut


def square_box(box):
    ymin = box[1]
    ymax = box[3]
    
    xmin = box[0]
    xmax = box[2]
    
    height = ymax-ymin
    width = xmax-xmin

    if (height>width):
        center = (xmax+xmin)/2
        xmin = center-(height/2)
        xmax = center+(height/2)
    else:
        center = (ymax+ymin)/2
        ymin = center-(width/2)
        ymax = center+(width/2)
    
    return np.array([xmin, ymin, xmax, ymax])

 
def bbox_from_landmark(landmarks, r_eye_corner_idx, l_eye_corner_idx):
    #start = time.perf_counter()
    mins = np.amin(landmarks, axis=0)
    maxes = np.amax(landmarks, axis=0)

    bbox = np.concatenate((mins, maxes))

    bbox_width  = bbox[2]-bbox[0]
    bbox_height = bbox[3]-bbox[1]

    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    breadth = max(bbox_width, bbox_height)
    SQRT2 =  1.4142135623730951
    new_breadth = breadth*SQRT2

    rise = landmarks[l_eye_corner_idx][1] - landmarks[r_eye_corner_idx][1]
    run = landmarks[l_eye_corner_idx][0] - landmarks[r_eye_corner_idx][0]

    face_roll_angle = atan2(rise, run)+pi/2
    angled_width = abs(cos(face_roll_angle))*new_breadth
    angled_height = abs(sin(face_roll_angle))*new_breadth
    
    X_SCALER = 1#(1+0.2)

    bbox_calc = np.empty(4)
    #bbox_calc[0] = center[0] - (bbox_width/2)
    #bbox_calc[1] = center[1] - (new_height/2)
    #bbox_calc[2] = bbox_calc[0] + bbox_width
    #bbox_calc[3] = bbox_calc[1] + new_height

    new_width  = max(bbox_width, angled_width)
    new_height = max(bbox_height, angled_height)

    bbox_calc[0] = center[0] - (new_width*X_SCALER/2)
    bbox_calc[1] = center[1] - (new_height/2)
    bbox_calc[2] = center[0] + (new_width*X_SCALER/2)
    bbox_calc[3] = center[1] + (new_height/2)

    #print("t = ",time.perf_counter()-start)
    
    return bbox_calc


def bbox_area(bbox):
    if bbox.size == 0:
        return 0.0
    else:
        return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])


def bbox_intersect(bbox_current, bbox_prev):
    if (bbox_current[0] > bbox_prev[2] or bbox_prev[0] > bbox_current[2]) \
        or (bbox_current[1] > bbox_prev[3] or bbox_prev[1] > bbox_current[3]):
        return np.empty(0)
    
    intersection = np.empty(4)
    intersection[0] = max(bbox_current[0], bbox_prev[0])
    intersection[1] = max(bbox_current[1], bbox_prev[1])
    intersection[2] = min(bbox_current[2], bbox_prev[2])
    intersection[3] = min(bbox_current[3], bbox_prev[3])
    return intersection


def bboxes_average(bbox1, bbox2):
    return np.average([bbox1, bbox2], axis=0)


def unwrap_bbox(bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
        
    return xmin, ymin, xmax, ymax


def crop_at_corners(bbox: np.ndarray, width: int, height: int):
    cropped_bbox = np.empty(4)

    cropped_bbox[0] = max(bbox[0], 0)
    cropped_bbox[1] = max(bbox[1], 0)
    cropped_bbox[2] = min(bbox[2], width-1)
    cropped_bbox[3] = min(bbox[3], height-1)

    return cropped_bbox